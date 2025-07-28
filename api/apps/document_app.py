#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License
#

"""
文档管理应用模块
提供文档上传、网页爬取、创建、列表、过滤、状态管理、删除、重命名、解析器更改等功能的API接口
"""

import json
import os.path
import pathlib
import re
from pathlib import Path

import flask
from flask import request
from flask_login import current_user, login_required

# 导入系统配置和常量
from api import settings
from api.constants import FILE_NAME_LEN_LIMIT, IMG_BASE64_PREFIX
from api.db import VALID_FILE_TYPES, VALID_TASK_STATUS, FileSource, FileType, ParserType, TaskStatus
from api.db.db_models import File, Task
from api.db.services import duplicate_name
from api.db.services.document_service import DocumentService, doc_upload_and_parse
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import TaskService, queue_tasks, cancel_all_task_of
from api.db.services.user_service import UserTenantService
from api.utils import get_uuid
from api.utils.api_utils import (
    get_data_error_result,
    get_json_result,
    server_error_response,
    validate_request,
)
from api.utils.file_utils import filename_type, get_project_base_directory, thumbnail
from api.utils.web_utils import CONTENT_TYPE_MAP, html2pdf, is_valid_url
from deepdoc.parser.html_parser import RAGFlowHtmlParser
from rag.nlp import search
from rag.utils.storage_factory import STORAGE_IMPL


@manager.route("/upload", methods=["POST"])  # noqa: F821
@login_required
@validate_request("kb_id")
def upload():
    """
    文档上传接口
    支持多文件上传到指定知识库，自动处理文件解析和存储
    
    Returns:
        JSON响应：包含上传成功的文档信息列表
    """
    # 获取知识库ID
    kb_id = request.form.get("kb_id")
    if not kb_id:
        return get_json_result(data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR)
    
    # 检查是否有文件上传
    if "file" not in request.files:
        return get_json_result(data=False, message="No file part!", code=settings.RetCode.ARGUMENT_ERROR)

    # 获取所有上传的文件对象
    file_objs = request.files.getlist("file")
    
    # 验证每个文件的文件名
    for file_obj in file_objs:
        if file_obj.filename == "":
            return get_json_result(data=False, message="No file selected!", code=settings.RetCode.ARGUMENT_ERROR)
        # 检查文件名长度限制
        if len(file_obj.filename.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
            return get_json_result(data=False, message=f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.", code=settings.RetCode.ARGUMENT_ERROR)

    # 验证知识库是否存在
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    if not e:
        raise LookupError("Can't find this knowledgebase!")
    
    # 调用文件服务上传文档
    err, files = FileService.upload_document(kb, file_objs, current_user.id)

    # 处理上传错误
    if err:
        return get_json_result(data=files, message="\n".join(err), code=settings.RetCode.SERVER_ERROR)

    # 检查是否有成功上传的文件
    if not files:
        return get_json_result(data=files, message="There seems to be an issue with your file format. Please verify it is correct and not corrupted.", code=settings.RetCode.DATA_ERROR)
    
    # 移除blob数据，只返回文件信息
    files = [f[0] for f in files]  # remove the blob

    return get_json_result(data=files)


@manager.route("/web_crawl", methods=["POST"])  # noqa: F821
@login_required
@validate_request("kb_id", "name", "url")
def web_crawl():
    """
    网页爬取接口
    将指定URL的网页内容转换为PDF文档并存储到知识库
    
    Returns:
        JSON响应：爬取操作结果
    """
    # 获取请求参数
    kb_id = request.form.get("kb_id")
    if not kb_id:
        return get_json_result(data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR)
    name = request.form.get("name")
    url = request.form.get("url")
    
    # 验证URL格式
    if not is_valid_url(url):
        return get_json_result(data=False, message="The URL format is invalid", code=settings.RetCode.ARGUMENT_ERROR)
    
    # 验证知识库是否存在
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    if not e:
        raise LookupError("Can't find this knowledgebase!")

    # 将网页转换为PDF
    blob = html2pdf(url)
    if not blob:
        return server_error_response(ValueError("Download failure."))

    # 获取根文件夹并初始化知识库文档结构
    root_folder = FileService.get_root_folder(current_user.id)
    pf_id = root_folder["id"]
    FileService.init_knowledgebase_docs(pf_id, current_user.id)
    kb_root_folder = FileService.get_kb_folder(current_user.id)
    kb_folder = FileService.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

    try:
        # 处理文件名重复
        filename = duplicate_name(DocumentService.query, name=name + ".pdf", kb_id=kb.id)
        filetype = filename_type(filename)
        if filetype == FileType.OTHER.value:
            raise RuntimeError("This type of file has not been supported yet!")

        # 确定存储位置
        location = filename
        while STORAGE_IMPL.obj_exist(kb_id, location):
            location += "_"
        STORAGE_IMPL.put(kb_id, location, blob)
        
        # 创建文档记录
        doc = {
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "parser_config": kb.parser_config,
            "created_by": current_user.id,
            "type": filetype,
            "name": filename,
            "location": location,
            "size": len(blob),
            "thumbnail": thumbnail(filename, blob),
            "suffix": Path(filename).suffix.lstrip("."),
        }
        
        # 根据文件类型设置解析器
        if doc["type"] == FileType.VISUAL:
            doc["parser_id"] = ParserType.PICTURE.value
        if doc["type"] == FileType.AURAL:
            doc["parser_id"] = ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            doc["parser_id"] = ParserType.PRESENTATION.value
        if re.search(r"\.(eml)$", filename):
            doc["parser_id"] = ParserType.EMAIL.value
        
        # 保存文档记录
        DocumentService.insert(doc)
        FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
    except Exception as e:
        return server_error_response(e)
    return get_json_result(data=True)


@manager.route("/create", methods=["POST"])  # noqa: F821
@login_required
@validate_request("name", "kb_id")
def create():
    """
    创建虚拟文档接口
    在指定知识库中创建一个虚拟文档（无实际文件内容）
    
    Returns:
        JSON响应：包含创建成功的文档信息
    """
    req = request.json
    kb_id = req["kb_id"]
    if not kb_id:
        return get_json_result(data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR)
    
    # 检查文件名长度限制
    if len(req["name"].encode("utf-8")) > FILE_NAME_LEN_LIMIT:
        return get_json_result(data=False, message=f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.", code=settings.RetCode.ARGUMENT_ERROR)

    # 检查文件名是否为空
    if req["name"].strip() == "":
        return get_json_result(data=False, message="File name can't be empty.", code=settings.RetCode.ARGUMENT_ERROR)
    req["name"] = req["name"].strip()

    try:
        # 验证知识库是否存在
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            return get_data_error_result(message="Can't find this knowledgebase!")

        # 检查文档名称是否重复
        if DocumentService.query(name=req["name"], kb_id=kb_id):
            return get_data_error_result(message="Duplicated document name in the same knowledgebase.")

        # 创建虚拟文档记录
        doc = DocumentService.insert(
            {
                "id": get_uuid(),
                "kb_id": kb.id,
                "parser_id": kb.parser_id,
                "parser_config": kb.parser_config,
                "created_by": current_user.id,
                "type": FileType.VIRTUAL,
                "name": req["name"],
                "suffix": Path(req["name"]).suffix.lstrip("."),
                "location": "",
                "size": 0,
            }
        )
        return get_json_result(data=doc.to_json())
    except Exception as e:
        return server_error_response(e)


@manager.route("/list", methods=["POST"])  # noqa: F821
@login_required
def list_docs():
    """
    获取文档列表接口
    支持分页、排序、关键词搜索、状态过滤、类型过滤等
    
    Returns:
        JSON响应：包含文档列表、总数和缩略图信息
    """
    # 获取知识库ID
    kb_id = request.args.get("kb_id")
    if not kb_id:
        return get_json_result(data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR)
    
    # 验证用户是否有权限访问该知识库
    tenants = UserTenantService.query(user_id=current_user.id)
    for tenant in tenants:
        if KnowledgebaseService.query(tenant_id=tenant.tenant_id, id=kb_id):
            break
    else:
        return get_json_result(data=False, message="Only owner of knowledgebase authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)
    
    # 获取查询参数
    keywords = request.args.get("keywords", "")
    page_number = int(request.args.get("page", 0))
    items_per_page = int(request.args.get("page_size", 0))
    orderby = request.args.get("orderby", "create_time")
    if request.args.get("desc", "true").lower() == "false":
        desc = False
    else:
        desc = True

    # 获取过滤条件
    req = request.get_json()

    # 验证运行状态过滤条件
    run_status = req.get("run_status", [])
    if run_status:
        invalid_status = {s for s in run_status if s not in VALID_TASK_STATUS}
        if invalid_status:
            return get_data_error_result(message=f"Invalid filter run status conditions: {', '.join(invalid_status)}")

    # 验证文件类型过滤条件
    types = req.get("types", [])
    if types:
        invalid_types = {t for t in types if t not in VALID_FILE_TYPES}
        if invalid_types:
            return get_data_error_result(message=f"Invalid filter conditions: {', '.join(invalid_types)} type{'s' if len(invalid_types) > 1 else ''}")

    # 获取文件后缀过滤条件
    suffix = req.get("suffix", [])

    try:
        # 获取文档列表和总数
        docs, tol = DocumentService.get_by_kb_id(kb_id, page_number, items_per_page, orderby, desc, keywords, run_status, types, suffix)

        # 处理缩略图URL
        for doc_item in docs:
            if doc_item["thumbnail"] and not doc_item["thumbnail"].startswith(IMG_BASE64_PREFIX):
                doc_item["thumbnail"] = f"/v1/document/image/{kb_id}-{doc_item['thumbnail']}"

        return get_json_result(data={"total": tol, "docs": docs})
    except Exception as e:
        return server_error_response(e)


@manager.route("/filter", methods=["POST"])  # noqa: F821
@login_required
def get_filter():
    """
    获取文档过滤条件接口
    根据过滤条件获取符合条件的文档统计信息
    
    Returns:
        JSON响应：包含过滤结果和总数
    """
    req = request.get_json()

    # 获取知识库ID
    kb_id = req.get("kb_id")
    if not kb_id:
        return get_json_result(data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR)
    
    # 验证用户权限
    tenants = UserTenantService.query(user_id=current_user.id)
    for tenant in tenants:
        if KnowledgebaseService.query(tenant_id=tenant.tenant_id, id=kb_id):
            break
    else:
        return get_json_result(data=False, message="Only owner of knowledgebase authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)

    # 获取过滤参数
    keywords = req.get("keywords", "")
    suffix = req.get("suffix", [])

    # 验证运行状态过滤条件
    run_status = req.get("run_status", [])
    if run_status:
        invalid_status = {s for s in run_status if s not in VALID_TASK_STATUS}
        if invalid_status:
            return get_data_error_result(message=f"Invalid filter run status conditions: {', '.join(invalid_status)}")

    # 验证文件类型过滤条件
    types = req.get("types", [])
    if types:
        invalid_types = {t for t in types if t not in VALID_FILE_TYPES}
        if invalid_types:
            return get_data_error_result(message=f"Invalid filter conditions: {', '.join(invalid_types)} type{'s' if len(invalid_types) > 1 else ''}")

    try:
        # 获取过滤结果
        filter, total = DocumentService.get_filter_by_kb_id(kb_id, keywords, run_status, types, suffix)
        return get_json_result(data={"total": total, "filter": filter})
    except Exception as e:
        return server_error_response(e)


@manager.route("/infos", methods=["POST"])  # noqa: F821
@login_required
def docinfos():
    """
    获取文档详细信息接口
    批量获取指定文档的详细信息
    
    Returns:
        JSON响应：包含文档详细信息列表
    """
    req = request.json
    doc_ids = req["doc_ids"]
    
    # 验证用户是否有权限访问所有文档
    for doc_id in doc_ids:
        if not DocumentService.accessible(doc_id, current_user.id):
            return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)
    
    # 获取文档详细信息
    docs = DocumentService.get_by_ids(doc_ids)
    return get_json_result(data=list(docs.dicts()))


@manager.route("/thumbnails", methods=["GET"])  # noqa: F821
# @login_required
def thumbnails():
    """
    获取文档缩略图接口
    批量获取指定文档的缩略图信息
    
    Returns:
        JSON响应：包含文档ID和缩略图URL的映射
    """
    # 获取文档ID列表
    doc_ids = request.args.get("doc_ids").split(",")
    if not doc_ids:
        return get_json_result(data=False, message='Lack of "Document ID"', code=settings.RetCode.ARGUMENT_ERROR)

    try:
        # 获取缩略图信息
        docs = DocumentService.get_thumbnails(doc_ids)

        # 处理缩略图URL
        for doc_item in docs:
            if doc_item["thumbnail"] and not doc_item["thumbnail"].startswith(IMG_BASE64_PREFIX):
                doc_item["thumbnail"] = f"/v1/document/image/{doc_item['kb_id']}-{doc_item['thumbnail']}"

        # 返回文档ID和缩略图的映射
        return get_json_result(data={d["id"]: d["thumbnail"] for d in docs})
    except Exception as e:
        return server_error_response(e)


@manager.route("/change_status", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_ids", "status")
def change_status():
    """
    更改文档状态接口
    批量更改文档的可用状态（启用/禁用）
    
    Returns:
        JSON响应：包含每个文档的状态更改结果
    """
    req = request.get_json()
    doc_ids = req.get("doc_ids", [])
    status = str(req.get("status", ""))

    # 验证状态值
    if status not in ["0", "1"]:
        return get_json_result(data=False, message='"Status" must be either 0 or 1!', code=settings.RetCode.ARGUMENT_ERROR)

    result = {}
    for doc_id in doc_ids:
        # 验证用户权限
        if not DocumentService.accessible(doc_id, current_user.id):
            result[doc_id] = {"error": "No authorization."}
            continue

        try:
            # 获取文档信息
            e, doc = DocumentService.get_by_id(doc_id)
            if not e:
                result[doc_id] = {"error": "No authorization."}
                continue
            
            # 获取知识库信息
            e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
            if not e:
                result[doc_id] = {"error": "Can't find this knowledgebase!"}
                continue
            
            # 更新文档状态
            if not DocumentService.update_by_id(doc_id, {"status": str(status)}):
                result[doc_id] = {"error": "Database error (Document update)!"}
                continue

            # 更新搜索引擎中的文档状态
            status_int = int(status)
            if not settings.docStoreConn.update({"doc_id": doc_id}, {"available_int": status_int}, search.index_name(kb.tenant_id), doc.kb_id):
                result[doc_id] = {"error": "Database error (docStore update)!"}
            result[doc_id] = {"status": status}
        except Exception as e:
            result[doc_id] = {"error": f"Internal server error: {str(e)}"}

    return get_json_result(data=result)


@manager.route("/rm", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_id")
def rm():
    """
    删除文档接口
    支持批量删除文档，包括文件存储、数据库记录和搜索引擎索引
    
    Returns:
        JSON响应：删除操作结果
    """
    req = request.json
    doc_ids = req["doc_id"]
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    # 验证用户删除权限
    for doc_id in doc_ids:
        if not DocumentService.accessible4deletion(doc_id, current_user.id):
            return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)

    # 初始化文件系统结构
    root_folder = FileService.get_root_folder(current_user.id)
    pf_id = root_folder["id"]
    FileService.init_knowledgebase_docs(pf_id, current_user.id)
    errors = ""
    kb_table_num_map = {}
    
    # 逐个删除文档
    for doc_id in doc_ids:
        try:
            # 获取文档信息
            e, doc = DocumentService.get_by_id(doc_id)
            if not e:
                return get_data_error_result(message="Document not found!")
            tenant_id = DocumentService.get_tenant_id(doc_id)
            if not tenant_id:
                return get_data_error_result(message="Tenant not found!")

            # 获取存储地址
            b, n = File2DocumentService.get_storage_address(doc_id=doc_id)

            # 删除相关任务
            TaskService.filter_delete([Task.doc_id == doc_id])
            
            # 删除文档记录
            if not DocumentService.remove_document(doc, tenant_id):
                return get_data_error_result(message="Database error (Document removal)!")

            # 删除关联的文件记录
            f2d = File2DocumentService.get_by_document_id(doc_id)
            deleted_file_count = 0
            if f2d:
                deleted_file_count = FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
            File2DocumentService.delete_by_document_id(doc_id)
            
            # 删除存储中的文件
            if deleted_file_count > 0:
                STORAGE_IMPL.rm(b, n)

            # 特殊处理表格类型文档
            doc_parser = doc.parser_id
            if doc_parser == ParserType.TABLE:
                kb_id = doc.kb_id
                if kb_id not in kb_table_num_map:
                    counts = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                    kb_table_num_map[kb_id] = counts
                kb_table_num_map[kb_id] -= 1
                if kb_table_num_map[kb_id] <= 0:
                    KnowledgebaseService.delete_field_map(kb_id)
        except Exception as e:
            errors += str(e)

    if errors:
        return get_json_result(data=False, message=errors, code=settings.RetCode.SERVER_ERROR)

    return get_json_result(data=True)


@manager.route("/run", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_ids", "run")
def run():
    """
    运行文档解析接口
    控制文档的解析任务状态（开始、停止、取消等）
    
    Returns:
        JSON响应：运行操作结果
    """
    req = request.json
    
    # 验证用户权限
    for doc_id in req["doc_ids"]:
        if not DocumentService.accessible(doc_id, current_user.id):
            return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)
    
    try:
        kb_table_num_map = {}
        for id in req["doc_ids"]:
            # 准备任务信息
            info = {"run": str(req["run"]), "progress": 0}
            if str(req["run"]) == TaskStatus.RUNNING.value and req.get("delete", False):
                info["progress_msg"] = ""
                info["chunk_num"] = 0
                info["token_num"] = 0

            # 获取文档信息
            e, doc = DocumentService.get_by_id(id)
            if not e:
                return get_data_error_result(message="Document not found!")
            
            # 如果文档已完成，清除之前的统计信息
            if doc.run == TaskStatus.DONE.value:
                DocumentService.clear_chunk_num_when_rerun(doc.id)

            # 更新文档状态
            DocumentService.update_by_id(id, info)
            tenant_id = DocumentService.get_tenant_id(id)
            if not tenant_id:
                return get_data_error_result(message="Tenant not found!")
            
            # 重新获取文档信息
            e, doc = DocumentService.get_by_id(id)
            if not e:
                return get_data_error_result(message="Document not found!")
            
            # 如果需要删除，清除相关数据
            if req.get("delete", False):
                TaskService.filter_delete([Task.doc_id == id])
                if settings.docStoreConn.indexExist(search.index_name(tenant_id), doc.kb_id):
                    settings.docStoreConn.delete({"doc_id": id}, search.index_name(tenant_id), doc.kb_id)

            # 处理取消任务
            if str(req["run"]) == TaskStatus.CANCEL.value:
                cancel_all_task_of(id)

            # 处理开始解析任务
            if str(req["run"]) == TaskStatus.RUNNING.value:
                e, doc = DocumentService.get_by_id(id)
                doc = doc.to_dict()
                doc["tenant_id"] = tenant_id

                # 特殊处理表格类型文档
                doc_parser = doc.get("parser_id", ParserType.NAIVE)
                if doc_parser == ParserType.TABLE:
                    kb_id = doc.get("kb_id")
                    if not kb_id:
                        continue
                    if kb_id not in kb_table_num_map:
                        count = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                        kb_table_num_map[kb_id] = count
                        if kb_table_num_map[kb_id] <= 0:
                            KnowledgebaseService.delete_field_map(kb_id)
                
                # 获取存储地址并排队任务
                bucket, name = File2DocumentService.get_storage_address(doc_id=doc["id"])
                queue_tasks(doc, bucket, name, 0)

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/rename", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_id", "name")
def rename():
    """
    重命名文档接口
    支持文档重命名，但不允许更改文件扩展名
    
    Returns:
        JSON响应：重命名操作结果
    """
    req = request.json
    
    # 验证用户权限
    if not DocumentService.accessible(req["doc_id"], current_user.id):
        return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)
    
    try:
        # 获取文档信息
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        
        # 检查是否尝试更改文件扩展名
        if pathlib.Path(req["name"].lower()).suffix != pathlib.Path(doc.name.lower()).suffix:
            return get_json_result(data=False, message="The extension of file can't be changed", code=settings.RetCode.ARGUMENT_ERROR)
        
        # 检查文件名长度限制
        if len(req["name"].encode("utf-8")) > FILE_NAME_LEN_LIMIT:
            return get_json_result(data=False, message=f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.", code=settings.RetCode.ARGUMENT_ERROR)

        # 检查新名称是否与同级文档重复
        for d in DocumentService.query(name=req["name"], kb_id=doc.kb_id):
            if d.name == req["name"]:
                return get_data_error_result(message="Duplicated document name in the same knowledgebase.")

        # 更新文档名称
        if not DocumentService.update_by_id(req["doc_id"], {"name": req["name"]}):
            return get_data_error_result(message="Database error (Document rename)!")

        # 同步更新关联的文件名称
        informs = File2DocumentService.get_by_document_id(req["doc_id"])
        if informs:
            e, file = FileService.get_by_id(informs[0].file_id)
            FileService.update_by_id(file.id, {"name": req["name"]})

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/get/<doc_id>", methods=["GET"])  # noqa: F821
# @login_required
def get(doc_id):
    """
    获取文档内容接口
    返回文档的二进制内容，设置正确的Content-Type
    
    Args:
        doc_id: 文档ID
        
    Returns:
        Flask响应：包含文档内容和正确的Content-Type头
    """
    try:
        # 验证文档是否存在
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(message="Document not found!")

        # 获取存储地址和文档内容
        b, n = File2DocumentService.get_storage_address(doc_id=doc_id)
        response = flask.make_response(STORAGE_IMPL.get(b, n))

        # 设置Content-Type
        ext = re.search(r"\.([^.]+)$", doc.name.lower())
        ext = ext.group(1) if ext else None
        if ext:
            if doc.type == FileType.VISUAL.value:
                content_type = CONTENT_TYPE_MAP.get(ext, f"image/{ext}")
            else:
                content_type = CONTENT_TYPE_MAP.get(ext, f"application/{ext}")
            response.headers.set("Content-Type", content_type)
        return response
    except Exception as e:
        return server_error_response(e)


@manager.route("/change_parser", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_id", "parser_id")
def change_parser():
    """
    更改文档解析器接口
    更改文档的解析器类型，并重新处理文档内容
    
    Returns:
        JSON响应：解析器更改操作结果
    """
    req = request.json

    # 验证用户权限
    if not DocumentService.accessible(req["doc_id"], current_user.id):
        return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)
    
    try:
        # 获取文档信息
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        
        # 检查解析器是否相同
        if doc.parser_id.lower() == req["parser_id"].lower():
            if "parser_config" in req:
                if req["parser_config"] == doc.parser_config:
                    return get_json_result(data=True)
            else:
                return get_json_result(data=True)

        # 检查不支持的解析器组合
        if (doc.type == FileType.VISUAL and req["parser_id"] != "picture") or (re.search(r"\.(ppt|pptx|pages)$", doc.name) and req["parser_id"] != "presentation"):
            return get_data_error_result(message="Not supported yet!")

        # 更新解析器并重置进度
        e = DocumentService.update_by_id(doc.id, {"parser_id": req["parser_id"], "progress": 0, "progress_msg": "", "run": TaskStatus.UNSTART.value})
        if not e:
            return get_data_error_result(message="Document not found!")
        
        # 更新解析器配置
        if "parser_config" in req:
            DocumentService.update_parser_config(doc.id, req["parser_config"])
        
        # 如果文档已有处理结果，清除并重新处理
        if doc.token_num > 0:
            e = DocumentService.increment_chunk_num(doc.id, doc.kb_id, doc.token_num * -1, doc.chunk_num * -1, doc.process_duration * -1)
            if not e:
                return get_data_error_result(message="Document not found!")
            
            # 删除搜索引擎中的索引
            tenant_id = DocumentService.get_tenant_id(req["doc_id"])
            if not tenant_id:
                return get_data_error_result(message="Tenant not found!")
            if settings.docStoreConn.indexExist(search.index_name(tenant_id), doc.kb_id):
                settings.docStoreConn.delete({"doc_id": doc.id}, search.index_name(tenant_id), doc.kb_id)

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/image/<image_id>", methods=["GET"])  # noqa: F821
# @login_required
def get_image(image_id):
    """
    获取图片内容接口
    返回指定图片的二进制内容
    
    Args:
        image_id: 图片ID（格式：bucket-name）
        
    Returns:
        Flask响应：包含图片内容和image/JPEG Content-Type
    """
    try:
        # 解析图片ID
        arr = image_id.split("-")
        if len(arr) != 2:
            return get_data_error_result(message="Image not found.")
        bkt, nm = image_id.split("-")
        
        # 获取图片内容并设置响应
        response = flask.make_response(STORAGE_IMPL.get(bkt, nm))
        response.headers.set("Content-Type", "image/JPEG")
        return response
    except Exception as e:
        return server_error_response(e)


@manager.route("/upload_and_parse", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id")
def upload_and_parse():
    """
    上传并解析文档接口
    用于对话场景中的文档上传和即时解析
    
    Returns:
        JSON响应：包含解析后的文档ID列表
    """
    # 检查是否有文件上传
    if "file" not in request.files:
        return get_json_result(data=False, message="No file part!", code=settings.RetCode.ARGUMENT_ERROR)

    # Restfule接口中file_obj内容公式
    # file_obj.filename    # 文件名，如 "document.pdf"
    # file_obj.content_type # MIME类型，如 "application/pdf"
    # file_obj.headers     # HTTP头部信息
    # file_obj.stream      # 文件流对象
    file_objs = request.files.getlist("file")
    
    # 验证文件选择
    for file_obj in file_objs:
        if file_obj.filename == "":
            return get_json_result(data=False, message="No file selected!", code=settings.RetCode.ARGUMENT_ERROR)

    # conversation_id 当前会话的唯一标识
    doc_ids = doc_upload_and_parse(request.form.get("conversation_id"), file_objs, current_user.id)

    return get_json_result(data=doc_ids)


@manager.route("/parse", methods=["POST"])  # noqa: F821
@login_required
def parse():
    """
    解析文档接口
    支持URL解析和文件解析，返回解析后的文本内容
    
    Returns:
        JSON响应：包含解析后的文本内容
    """
    # 处理URL解析
    url = request.json.get("url") if request.json else ""
    if url:
        # 验证URL格式
        if not is_valid_url(url):
            return get_json_result(data=False, message="The URL format is invalid", code=settings.RetCode.ARGUMENT_ERROR)
        
        # 设置下载路径
        download_path = os.path.join(get_project_base_directory(), "logs/downloads")
        os.makedirs(download_path, exist_ok=True)
        
        # 使用Selenium进行网页抓取
        from seleniumwire.webdriver import Chrome, ChromeOptions

        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("prefs", {"download.default_directory": download_path, "download.prompt_for_download": False, "download.directory_upgrade": True, "safebrowsing.enabled": True})
        driver = Chrome(options=options)
        driver.get(url)
        res_headers = [r.response.headers for r in driver.requests if r and r.response]
        
        # 如果响应头数量大于1，直接解析HTML
        if len(res_headers) > 1:
            sections = RAGFlowHtmlParser().parser_txt(driver.page_source)
            driver.quit()
            return get_json_result(data="\n".join(sections))

        # 定义文件类用于处理下载的文件
        class File:
            filename: str
            filepath: str

            def __init__(self, filename, filepath):
                self.filename = filename
                self.filepath = filepath

            def read(self):
                with open(self.filepath, "rb") as f:
                    return f.read()

        # 从响应头中提取文件名
        r = re.search(r"filename=\"([^\"]+)\"", str(res_headers))
        if not r or not r.group(1):
            return get_json_result(data=False, message="Can't not identify downloaded file", code=settings.RetCode.ARGUMENT_ERROR)
        
        # 解析下载的文件
        f = File(r.group(1), os.path.join(download_path, r.group(1)))
        txt = FileService.parse_docs([f], current_user.id)
        return get_json_result(data=txt)

    # 处理文件解析
    if "file" not in request.files:
        return get_json_result(data=False, message="No file part!", code=settings.RetCode.ARGUMENT_ERROR)

    file_objs = request.files.getlist("file")
    txt = FileService.parse_docs(file_objs, current_user.id)

    return get_json_result(data=txt)


@manager.route("/set_meta", methods=["POST"])  # noqa: F821
@login_required
@validate_request("doc_id", "meta")
def set_meta():
    """
    设置文档元数据接口
    为文档设置自定义的元数据字段
    
    Returns:
        JSON响应：元数据设置操作结果
    """
    req = request.json
    
    # 验证用户权限
    if not DocumentService.accessible(req["doc_id"], current_user.id):
        return get_json_result(data=False, message="No authorization.", code=settings.RetCode.AUTHENTICATION_ERROR)
    
    try:
        # 解析JSON格式的元数据
        meta = json.loads(req["meta"])
    except Exception as e:
        return get_json_result(data=False, message=f"Json syntax error: {e}", code=settings.RetCode.ARGUMENT_ERROR)
    
    # 验证元数据格式
    if not isinstance(meta, dict):
        return get_json_result(data=False, message='Meta data should be in Json map format, like {"key": "value"}', code=settings.RetCode.ARGUMENT_ERROR)

    try:
        # 获取文档信息
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")

        # 更新文档元数据
        if not DocumentService.update_by_id(req["doc_id"], {"meta_fields": meta}):
            return get_data_error_result(message="Database error (meta updates)!")

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
