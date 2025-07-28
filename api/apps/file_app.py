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
文件管理应用模块
提供文件上传、创建、列表、删除、重命名、移动等功能的API接口
"""

import os
import pathlib
import re

import flask
from flask import request
from flask_login import login_required, current_user

# 导入数据库服务模块
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.utils import get_uuid
from api.db import FileType, FileSource
from api.db.services import duplicate_name
from api.db.services.file_service import FileService
from api import settings
from api.utils.api_utils import get_json_result
from api.utils.file_utils import filename_type
from api.utils.web_utils import CONTENT_TYPE_MAP
from rag.utils.storage_factory import STORAGE_IMPL


@manager.route('/upload', methods=['POST'])  # noqa: F821
@login_required
# @validate_request("parent_id")
def upload():
    """
    文件上传接口
    支持多文件上传，自动创建文件夹结构，处理文件重名
    
    Returns:
        JSON响应：包含上传成功的文件信息列表
    """
    # 获取父文件夹ID，如果未提供则使用根文件夹
    pf_id = request.form.get("parent_id")

    if not pf_id:
        root_folder = FileService.get_root_folder(current_user.id)
        pf_id = root_folder["id"]

    # 检查是否有文件上传
    if 'file' not in request.files:
        return get_json_result(
            data=False, message='No file part!', code=settings.RetCode.ARGUMENT_ERROR)
    file_objs = request.files.getlist('file')

    # 验证文件名称是否为空
    for file_obj in file_objs:
        if file_obj.filename == '':
            return get_json_result(
                data=False, message='No file selected!', code=settings.RetCode.ARGUMENT_ERROR)
    
    file_res = []
    try:
        # 获取父文件夹信息
        e, pf_folder = FileService.get_by_id(pf_id)
        if not e:
            return get_data_error_result( message="Can't find this folder!")
        
        # 处理每个上传的文件
        for file_obj in file_objs:
            # 检查用户文件数量限制
            MAX_FILE_NUM_PER_USER = int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))
            if MAX_FILE_NUM_PER_USER > 0 and DocumentService.get_doc_count(current_user.id) >= MAX_FILE_NUM_PER_USER:
                return get_data_error_result( message="Exceed the maximum file number of a free user!")

            # 分割文件路径，获取文件夹结构
            if not file_obj.filename:
                file_obj_names = [pf_folder.name, file_obj.filename]
            else:
                full_path = '/' + file_obj.filename
                file_obj_names = full_path.split('/')
            file_len = len(file_obj_names)

            # 获取文件夹ID列表，用于创建文件夹结构
            file_id_list = FileService.get_id_list_by_id(pf_id, file_obj_names, 1, [pf_id])
            len_id_list = len(file_id_list)

            # 创建缺失的文件夹结构
            if file_len != len_id_list:
                e, file = FileService.get_by_id(file_id_list[len_id_list - 1])
                if not e:
                    return get_data_error_result(message="Folder not found!")
                last_folder = FileService.create_folder(file, file_id_list[len_id_list - 1], file_obj_names,
                                                        len_id_list)
            else:
                e, file = FileService.get_by_id(file_id_list[len_id_list - 2])
                if not e:
                    return get_data_error_result(message="Folder not found!")
                last_folder = FileService.create_folder(file, file_id_list[len_id_list - 2], file_obj_names,
                                                        len_id_list)

            # 根据文件后缀确定文件路径
            filetype = filename_type(file_obj_names[file_len - 1])
            # 确定文件存储位置，就是文件名
            location = file_obj_names[file_len - 1]
            # 处理文件名冲突，自动添加后缀
            while STORAGE_IMPL.obj_exist(last_folder.id, location):
                location += "_"
            
            # 读取文件内容并存储
            blob = file_obj.read()
            # 基于父目录ID，产出是否已存在同名文件记录
            filename = duplicate_name(
                FileService.query,
                name=file_obj_names[file_len - 1],
                parent_id=last_folder.id)
            #文件存储入文件系统中
            STORAGE_IMPL.put(last_folder.id, location, blob)
            
            # 创建文件记录
            file = {
                "id": get_uuid(),
                "parent_id": last_folder.id,
                "tenant_id": current_user.id,
                "created_by": current_user.id,
                "type": filetype,
                "name": filename,
                "location": location,
                "size": len(blob),
            }
            # 文件信息记录入数据库中
            file = FileService.insert(file)
            file_res.append(file.to_json())
        return get_json_result(data=file_res)
    except Exception as e:
        return server_error_response(e)


@manager.route('/create', methods=['POST'])  # noqa: F821
@login_required
@validate_request("name")
def create():
    """
    创建文件夹或虚拟文件接口
    
    Returns:
        JSON响应：包含创建成功的文件信息
    """
    req = request.json
    pf_id = request.json.get("parent_id")
    input_file_type = request.json.get("type")
    
    # 如果未提供父文件夹ID，使用根文件夹
    if not pf_id:
        root_folder = FileService.get_root_folder(current_user.id)
        pf_id = root_folder["id"]

    try:
        # 验证父文件夹是否存在
        if not FileService.is_parent_folder_exist(pf_id):
            return get_json_result(
                data=False, message="Parent Folder Doesn't Exist!", code=settings.RetCode.OPERATING_ERROR)
        
        # 检查文件夹名称是否重复
        if FileService.query(name=req["name"], parent_id=pf_id):
            return get_data_error_result(
                message="Duplicated folder name in the same folder.")

        # 确定文件类型
        if input_file_type == FileType.FOLDER.value:
            file_type = FileType.FOLDER.value
        else:
            file_type = FileType.VIRTUAL.value

        # 文件夹信息记录入数据库中
        file = FileService.insert({
            "id": get_uuid(),
            "parent_id": pf_id,
            "tenant_id": current_user.id,
            "created_by": current_user.id,
            "name": req["name"],
            "location": "",
            "size": 0,
            "type": file_type
        })

        return get_json_result(data=file.to_json())
    except Exception as e:
        return server_error_response(e)


@manager.route('/list', methods=['GET'])  # noqa: F821
@login_required
def list_files():
    """
    获取文件列表接口
    支持分页、排序、关键词搜索
    
    Returns:
        JSON响应：包含文件列表、总数和父文件夹信息
    """
    pf_id = request.args.get("parent_id")

    # 获取查询参数
    keywords = request.args.get("keywords", "")
    page_number = int(request.args.get("page", 1))
    items_per_page = int(request.args.get("page_size", 15))
    orderby = request.args.get("orderby", "create_time")
    desc = request.args.get("desc", True)
    
    # 如果未提供父文件夹ID，使用根文件夹
    if not pf_id:
        root_folder = FileService.get_root_folder(current_user.id)
        pf_id = root_folder["id"]
        FileService.init_knowledgebase_docs(pf_id, current_user.id)
    
    try:
        # 验证文件夹是否存在
        e, file = FileService.get_by_id(pf_id)
        if not e:
            return get_data_error_result(message="Folder not found!")

        # 获取文件列表和总数
        files, total = FileService.get_by_pf_id(
            current_user.id, pf_id, page_number, items_per_page, orderby, desc, keywords)

        # 获取父文件夹信息
        parent_folder = FileService.get_parent_folder(pf_id)
        if not parent_folder:
            return get_json_result(message="File not found!")

        return get_json_result(data={"total": total, "files": files, "parent_folder": parent_folder.to_json()})
    except Exception as e:
        return server_error_response(e)


@manager.route('/root_folder', methods=['GET'])  # noqa: F821
@login_required
def get_root_folder():
    """
    获取用户根文件夹接口
    
    Returns:
        JSON响应：包含根文件夹信息
    """
    try:
        root_folder = FileService.get_root_folder(current_user.id)
        return get_json_result(data={"root_folder": root_folder})
    except Exception as e:
        return server_error_response(e)

@manager.route('/parent_folder', methods=['GET'])  # noqa: F821
@login_required
def get_parent_folder():
    """
    获取指定文件的父文件夹接口
    
    Returns:
        JSON响应：包含父文件夹信息
    """
    file_id = request.args.get("file_id")
    try:
        # 验证文件是否存在
        e, file = FileService.get_by_id(file_id)
        if not e:
            return get_data_error_result(message="Folder not found!")

        # 获取父文件夹
        parent_folder = FileService.get_parent_folder(file_id)
        return get_json_result(data={"parent_folder": parent_folder.to_json()})
    except Exception as e:
        return server_error_response(e)


@manager.route('/all_parent_folder', methods=['GET'])  # noqa: F821
@login_required
def get_all_parent_folders():
    """
    获取指定文件的所有父文件夹路径接口
    
    Returns:
        JSON响应：包含所有父文件夹信息列表
    """
    file_id = request.args.get("file_id")
    try:
        # 验证文件是否存在
        e, file = FileService.get_by_id(file_id)
        if not e:
            return get_data_error_result(message="Folder not found!")

        # 获取所有父文件夹
        parent_folders = FileService.get_all_parent_folders(file_id)
        parent_folders_res = []
        for parent_folder in parent_folders:
            parent_folders_res.append(parent_folder.to_json())
        return get_json_result(data={"parent_folders": parent_folders_res})
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])  # noqa: F821
@login_required
@validate_request("file_ids")
def rm():
    """
    删除文件或文件夹接口
    支持批量删除，递归删除文件夹内容
    
    Returns:
        JSON响应：删除操作结果
    """
    req = request.json
    file_ids = req["file_ids"]
    try:
        for file_id in file_ids:
            # 验证文件是否存在
            e, file = FileService.get_by_id(file_id)
            if not e:
                return get_data_error_result(message="File or Folder not found!")
            if not file.tenant_id:
                return get_data_error_result(message="Tenant not found!")
            
            # 跳过知识库源文件
            if file.source_type == FileSource.KNOWLEDGEBASE:
                continue

            # 根据文件类型执行不同的删除逻辑
            if file.type == FileType.FOLDER.value:
                # 文件夹：递归删除所有子文件
                file_id_list = FileService.get_all_innermost_file_ids(file_id, [])
                for inner_file_id in file_id_list:
                    e, file = FileService.get_by_id(inner_file_id)
                    if not e:
                        return get_data_error_result(message="File not found!")
                    # 删除目录下的文件的对象存储记录
                    STORAGE_IMPL.rm(file.parent_id, file.location)
                # 删除数据库记录
                FileService.delete_folder_by_pf_id(current_user.id, file_id)
            else:
                # 普通文件：直接删除对象存储
                STORAGE_IMPL.rm(file.parent_id, file.location)
                # 删除数据库记录
                if not FileService.delete(file):
                    return get_data_error_result(
                        message="Database error (File removal)!")

            # 删除关联的文档记录
            informs = File2DocumentService.get_by_file_id(file_id)
            for inform in informs:
                doc_id = inform.document_id
                e, doc = DocumentService.get_by_id(doc_id)
                if not e:
                    return get_data_error_result(message="Document not found!")
                tenant_id = DocumentService.get_tenant_id(doc_id)
                if not tenant_id:
                    return get_data_error_result(message="Tenant not found!")
                if not DocumentService.remove_document(doc, tenant_id):
                    return get_data_error_result(
                        message="Database error (Document removal)!")
            File2DocumentService.delete_by_file_id(file_id)

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/rename', methods=['POST'])  # noqa: F821
@login_required
@validate_request("file_id", "name")
def rename():
    """
    重命名文件或文件夹接口
    支持文件重命名，但不允许更改文件扩展名
    
    Returns:
        JSON响应：重命名操作结果
    """
    req = request.json
    try:
        # 验证文件是否存在
        e, file = FileService.get_by_id(req["file_id"])
        if not e:
            return get_data_error_result(message="File not found!")
        
        # 检查是否尝试更改文件扩展名
        if file.type != FileType.FOLDER.value \
            and pathlib.Path(req["name"].lower()).suffix != pathlib.Path(
                file.name.lower()).suffix:
            return get_json_result(
                data=False,
                message="The extension of file can't be changed",
                code=settings.RetCode.ARGUMENT_ERROR)
        
        # 检查新名称是否与同级文件重复
        for file in FileService.query(name=req["name"], pf_id=file.parent_id):
            if file.name == req["name"]:
                return get_data_error_result(
                    message="Duplicated file name in the same folder.")

        # 更新文件名称
        if not FileService.update_by_id(
                req["file_id"], {"name": req["name"]}):
            return get_data_error_result(
                message="Database error (File rename)!")
        
        # 应该添加：更新存储中的文件名
        # 获取旧的文件位置
        old_location = file.location
        # 构建新的文件位置（保持路径结构，只改文件名）
        new_location = req["name"]  # 或者根据路径结构构建新的location

        # 在存储中重命名文件
        STORAGE_IMPL.rename(file.parent_id, old_location, new_location)

        # 同步更新关联的文档名称
        informs = File2DocumentService.get_by_file_id(req["file_id"])
        if informs:
            if not DocumentService.update_by_id(
                    informs[0].document_id, {"name": req["name"]}):
                return get_data_error_result(
                    message="Database error (Document rename)!")

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/get/<file_id>', methods=['GET'])  # noqa: F821
@login_required
def get(file_id):
    """
    获取文件内容接口
    返回文件的二进制内容，设置正确的Content-Type
    
    Args:
        file_id: 文件ID
        
    Returns:
        Flask响应：包含文件内容和正确的Content-Type头
    """
    try:
        # 验证文件是否存在
        e, file = FileService.get_by_id(file_id)
        if not e:
            return get_data_error_result(message="Document not found!")

        # 获取文件内容
        blob = STORAGE_IMPL.get(file.parent_id, file.location)
        if not blob:
            # 如果直接获取失败，尝试通过文件-文档关联获取
            b, n = File2DocumentService.get_storage_address(file_id=file_id)
            blob = STORAGE_IMPL.get(b, n)

        # 创建响应并设置Content-Type
        response = flask.make_response(blob)
        ext = re.search(r"\.([^.]+)$", file.name.lower())
        ext = ext.group(1) if ext else None
        if ext:
            if file.type == FileType.VISUAL.value:
                content_type = CONTENT_TYPE_MAP.get(ext, f"image/{ext}")
            else:
                content_type = CONTENT_TYPE_MAP.get(ext, f"application/{ext}")
            response.headers.set("Content-Type", content_type)
        return response
    except Exception as e:
        return server_error_response(e)


@manager.route('/mv', methods=['POST'])  # noqa: F821
@login_required
@validate_request("src_file_ids", "dest_file_id")
def move():
    """
    移动文件或文件夹接口
    支持批量移动文件到指定目标文件夹
    
    Returns:
        JSON响应：移动操作结果
    """
    req = request.json
    try:
        file_ids = req["src_file_ids"]
        parent_id = req["dest_file_id"]
        
        # 获取所有要移动的文件信息
        files = FileService.get_by_ids(file_ids)
        files_dict = {}
        for file in files:
            files_dict[file.id] = file

        # 验证所有文件是否存在且属于当前用户
        for file_id in file_ids:
            file = files_dict[file_id]
            if not file:
                return get_data_error_result(message="File or Folder not found!")
            if not file.tenant_id:
                return get_data_error_result(message="Tenant not found!")
        
        # 验证目标文件夹是否存在
        fe, _ = FileService.get_by_id(parent_id)
        if not fe:
            return get_data_error_result(message="Parent Folder not found!")
        
        # 执行移动操作
        FileService.move_file(file_ids, parent_id)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
