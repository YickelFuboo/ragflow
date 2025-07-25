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
#  limitations under the License.
#
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask_login import current_user
from peewee import fn

from api.constants import FILE_NAME_LEN_LIMIT
from api.db import KNOWLEDGEBASE_FOLDER_NAME, FileSource, FileType, ParserType
from api.db.db_models import DB, Document, File, File2Document, Knowledgebase
from api.db.services import duplicate_name
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.utils import get_uuid
from api.utils.file_utils import filename_type, read_potential_broken_pdf, thumbnail_img
from rag.utils.storage_factory import STORAGE_IMPL


class FileService(CommonService):
    # Service class for managing file operations and storage
    model = File

    @classmethod
    @DB.connection_context()
    def get_by_pf_id(cls, tenant_id, pf_id, page_number, items_per_page, orderby, desc, keywords):
        # Get files by parent folder ID with pagination and filtering
        # Args:
        #     tenant_id: ID of the tenant
        #     pf_id: Parent folder ID
        #     page_number: Page number for pagination
        #     items_per_page: Number of items per page
        #     orderby: Field to order by
        #     desc: Boolean indicating descending order
        #     keywords: Search keywords
        # Returns:
        #     Tuple of (file_list, total_count)
        if keywords:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == pf_id), (fn.LOWER(cls.model.name).contains(keywords.lower())), ~(cls.model.id == pf_id))
        else:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == pf_id), ~(cls.model.id == pf_id))
        count = files.count()
        if desc:
            files = files.order_by(cls.model.getter_by(orderby).desc())
        else:
            files = files.order_by(cls.model.getter_by(orderby).asc())

        files = files.paginate(page_number, items_per_page)

        res_files = list(files.dicts())
        for file in res_files:
            if file["type"] == FileType.FOLDER.value:
                file["size"] = cls.get_folder_size(file["id"])
                file["kbs_info"] = []
                children = list(
                    cls.model.select()
                    .where(
                        (cls.model.tenant_id == tenant_id),
                        (cls.model.parent_id == file["id"]),
                        ~(cls.model.id == file["id"]),
                    )
                    .dicts()
                )
                file["has_child_folder"] = any(value["type"] == FileType.FOLDER.value for value in children)
                continue
            kbs_info = cls.get_kb_id_by_file_id(file["id"])
            file["kbs_info"] = kbs_info

        return res_files, count

    @classmethod
    @DB.connection_context()
    def get_kb_id_by_file_id(cls, file_id):
        # Get knowledge base IDs associated with a file
        # Args:
        #     file_id: File ID
        # Returns:
        #     List of dictionaries containing knowledge base IDs and names
        kbs = (
            cls.model.select(*[Knowledgebase.id, Knowledgebase.name])
            .join(File2Document, on=(File2Document.file_id == file_id))
            .join(Document, on=(File2Document.document_id == Document.id))
            .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
            .where(cls.model.id == file_id)
        )
        if not kbs:
            return []
        kbs_info_list = []
        for kb in list(kbs.dicts()):
            kbs_info_list.append({"kb_id": kb["id"], "kb_name": kb["name"]})
        return kbs_info_list

    @classmethod
    @DB.connection_context()
    def get_by_pf_id_name(cls, id, name):
        # Get file by parent folder ID and name
        # Args:
        #     id: Parent folder ID
        #     name: File name
        # Returns:
        #     File object or None if not found
        file = cls.model.select().where((cls.model.parent_id == id) & (cls.model.name == name))
        if file.count():
            e, file = cls.get_by_id(file[0].id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
            return file
        return None

    @classmethod
    @DB.connection_context()
    def get_id_list_by_id(cls, id, name, count, res):
        # Recursively get list of file IDs by traversing folder structure
        # Args:
        #     id: Starting folder ID
        #     name: List of folder names to traverse
        #     count: Current depth in traversal
        #     res: List to store results
        # Returns:
        #     List of file IDs
        if count < len(name):
            file = cls.get_by_pf_id_name(id, name[count])
            if file:
                res.append(file.id)
                return cls.get_id_list_by_id(file.id, name, count + 1, res)
            else:
                return res
        else:
            return res

    @classmethod
    @DB.connection_context()
    def get_all_innermost_file_ids(cls, folder_id, result_ids):
        # Get IDs of all files in the deepest level of folders
        # Args:
        #     folder_id: Starting folder ID
        #     result_ids: List to store results
        # Returns:
        #     List of file IDs
        subfolders = cls.model.select().where(cls.model.parent_id == folder_id)
        if subfolders.exists():
            for subfolder in subfolders:
                cls.get_all_innermost_file_ids(subfolder.id, result_ids)
        else:
            result_ids.append(folder_id)
        return result_ids

    @classmethod
    @DB.connection_context()
    def create_folder(cls, file, parent_id, name, count):
        # Recursively create folder structure
        # Args:
        #     file: Current file object
        #     parent_id: Parent folder ID
        #     name: List of folder names to create
        #     count: Current depth in creation
        # Returns:
        #     Created file object
        if count > len(name) - 2:
            return file
        else:
            file = cls.insert(
                {"id": get_uuid(), "parent_id": parent_id, "tenant_id": current_user.id, "created_by": current_user.id, "name": name[count], "location": "", "size": 0, "type": FileType.FOLDER.value}
            )
            return cls.create_folder(file, file.id, name, count + 1)

    @classmethod
    @DB.connection_context()
    def is_parent_folder_exist(cls, parent_id):
        # Check if parent folder exists
        # Args:
        #     parent_id: Parent folder ID
        # Returns:
        #     Boolean indicating if folder exists
        parent_files = cls.model.select().where(cls.model.id == parent_id)
        if parent_files.count():
            return True
        cls.delete_folder_by_pf_id(parent_id)
        return False

    @classmethod
    @DB.connection_context()
    def get_root_folder(cls, tenant_id):
        # Get or create root folder for tenant
        # Args:
        #     tenant_id: Tenant ID
        # Returns:
        #     Root folder dictionary
        for file in cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)):
            return file.to_dict()

        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": file_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": "/",
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def get_kb_folder(cls, tenant_id):
        # Get knowledge base folder for tenant
        # Args:
        #     tenant_id: Tenant ID
        # Returns:
        #     Knowledge base folder dictionary
        for root in cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)):
            for folder in cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == root.id), (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)):
                return folder.to_dict()
        assert False, "Can't find the KB folder. Database init error."

    @classmethod
    @DB.connection_context()
    def new_a_file_from_kb(cls, tenant_id, name, parent_id, ty=FileType.FOLDER.value, size=0, location=""):
        # Create a new file from knowledge base
        # Args:
        #     tenant_id: Tenant ID
        #     name: File name
        #     parent_id: Parent folder ID
        #     ty: File type
        #     size: File size
        #     location: File location
        # Returns:
        #     Created file dictionary
        for file in cls.query(tenant_id=tenant_id, parent_id=parent_id, name=name):
            return file.to_dict()
        file = {
            "id": get_uuid(),
            "parent_id": parent_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": name,
            "type": ty,
            "size": size,
            "location": location,
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def init_knowledgebase_docs(cls, root_id, tenant_id):
        # Initialize knowledge base documents
        # Args:
        #     root_id: Root folder ID
        #     tenant_id: Tenant ID
        for _ in cls.model.select().where((cls.model.name == KNOWLEDGEBASE_FOLDER_NAME) & (cls.model.parent_id == root_id)):
            return
        folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)

        for kb in Knowledgebase.select(*[Knowledgebase.id, Knowledgebase.name]).where(Knowledgebase.tenant_id == tenant_id):
            kb_folder = cls.new_a_file_from_kb(tenant_id, kb.name, folder["id"])
            for doc in DocumentService.query(kb_id=kb.id):
                FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], tenant_id)

    @classmethod
    @DB.connection_context()
    def get_parent_folder(cls, file_id):
        # Get parent folder of a file
        # Args:
        #     file_id: File ID
        # Returns:
        #     Parent folder object
        file = cls.model.select().where(cls.model.id == file_id)
        if file.count():
            e, file = cls.get_by_id(file[0].parent_id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
        else:
            raise RuntimeError("Database error (File doesn't exist)!")
        return file

    @classmethod
    @DB.connection_context()
    def get_all_parent_folders(cls, start_id):
        # Get all parent folders in path
        # Args:
        #     start_id: Starting file ID
        # Returns:
        #     List of parent folder objects
        parent_folders = []
        current_id = start_id
        while current_id:
            e, file = cls.get_by_id(current_id)
            if file.parent_id != file.id and e:
                parent_folders.append(file)
                current_id = file.parent_id
            else:
                parent_folders.append(file)
                break
        return parent_folders

    @classmethod
    @DB.connection_context()
    def insert(cls, file):
        # Insert a new file record
        # Args:
        #     file: File data dictionary
        # Returns:
        #     Created file object
        if not cls.save(**file):
            raise RuntimeError("Database error (File)!")
        return File(**file)

    @classmethod
    @DB.connection_context()
    def delete(cls, file):
        #
        return cls.delete_by_id(file.id)

    @classmethod
    @DB.connection_context()
    def delete_by_pf_id(cls, folder_id):
        return cls.model.delete().where(cls.model.parent_id == folder_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_folder_by_pf_id(cls, user_id, folder_id):
        try:
            files = cls.model.select().where((cls.model.tenant_id == user_id) & (cls.model.parent_id == folder_id))
            for file in files:
                cls.delete_folder_by_pf_id(user_id, file.id)
            return (cls.model.delete().where((cls.model.tenant_id == user_id) & (cls.model.id == folder_id)).execute(),)
        except Exception:
            logging.exception("delete_folder_by_pf_id")
            raise RuntimeError("Database error (File retrieval)!")

    @classmethod
    @DB.connection_context()
    def get_file_count(cls, tenant_id):
        files = cls.model.select(cls.model.id).where(cls.model.tenant_id == tenant_id)
        return len(files)

    @classmethod
    @DB.connection_context()
    def get_folder_size(cls, folder_id):
        size = 0

        def dfs(parent_id):
            nonlocal size
            for f in cls.model.select(*[cls.model.id, cls.model.size, cls.model.type]).where(cls.model.parent_id == parent_id, cls.model.id != parent_id):
                size += f.size
                if f.type == FileType.FOLDER.value:
                    dfs(f.id)

        dfs(folder_id)
        return size

    @classmethod
    @DB.connection_context()
    def add_file_from_kb(cls, doc, kb_folder_id, tenant_id):
        for _ in File2DocumentService.get_by_document_id(doc["id"]):
            return
        file = {
            "id": get_uuid(),
            "parent_id": kb_folder_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": doc["name"],
            "type": doc["type"],
            "size": doc["size"],
            "location": doc["location"],
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        File2DocumentService.save(**{"id": get_uuid(), "file_id": file["id"], "document_id": doc["id"]})

    @classmethod
    @DB.connection_context()
    def move_file(cls, file_ids, folder_id):
        try:
            cls.filter_update((cls.model.id << file_ids,), {"parent_id": folder_id})
        except Exception:
            logging.exception("move_file")
            raise RuntimeError("Database error (File move)!")

    def upload_document(self, kb, file_objs, user_id):
        """
        上传文档到知识库的核心方法
        
        Args:
            kb: 知识库对象，包含知识库的基本信息（如ID、租户ID、解析器配置等）
            file_objs: 文件对象列表，用户上传的文件（werkzeug.datastructures.FileStorage对象）
            user_id: 用户ID，用于权限验证和文件夹管理
            
        Returns:
            tuple: (err, files)
                - err: 处理失败的文件错误信息列表
                - files: 成功处理的文件信息列表，每个元素为(doc_info, blob)元组
        """
        # 步骤1: 初始化文件夹结构
        # 获取用户的根文件夹，用于存储所有文件
        root_folder = self.get_root_folder(user_id)
        pf_id = root_folder["id"]
        
        # 初始化知识库文档文件夹，确保文件夹结构存在
        self.init_knowledgebase_docs(pf_id, user_id)
        
        # 获取知识库根文件夹
        kb_root_folder = self.get_kb_folder(user_id)
        
        # 为当前知识库创建新的文件夹，用于存储该知识库的文档
        kb_folder = self.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

        # 步骤2: 初始化结果变量
        err, files = [], []  # err存储错误信息，files存储成功处理的文件
        
        # 步骤3: 逐个处理上传的文件
        for file in file_objs:
            try:
                # 步骤3.1: 检查文件数量限制
                # 获取每个用户最大文件数量限制，0表示无限制
                MAX_FILE_NUM_PER_USER = int(os.environ.get("MAX_FILE_NUM_PER_USER", 0))
                if MAX_FILE_NUM_PER_USER > 0 and DocumentService.get_doc_count(kb.tenant_id) >= MAX_FILE_NUM_PER_USER:
                    raise RuntimeError("Exceed the maximum file number of a free user!")
                
                # 步骤3.2: 检查文件名长度限制
                # 确保文件名不超过字节长度限制
                if len(file.filename.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
                    raise RuntimeError(f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.")

                # 步骤3.3: 处理文件名重复
                # 如果存在同名文件，自动添加序号避免冲突
                filename = duplicate_name(DocumentService.query, name=file.filename, kb_id=kb.id)
                
                # 步骤3.4: 识别文件类型
                # 根据文件扩展名判断文件类型
                filetype = filename_type(filename)
                if filetype == FileType.OTHER.value:
                    raise RuntimeError("This type of file has not been supported yet!")

                # 步骤3.5: 生成存储位置
                # 使用文件名作为存储位置，如果已存在则添加下划线
                location = filename
                while STORAGE_IMPL.obj_exist(kb.id, location):
                    location += "_"

                # 步骤3.6: 读取文件内容
                # 读取文件的二进制内容
                blob = file.read()
                
                # 步骤3.7: 特殊处理PDF文件
                # 如果是PDF文件，尝试修复可能损坏的PDF
                if filetype == FileType.PDF.value:
                    blob = read_potential_broken_pdf(blob)
                
                # 步骤3.8: 存储文件到对象存储
                # 将文件内容存储到对象存储系统（如MinIO）
                STORAGE_IMPL.put(kb.id, location, blob)

                # 步骤3.9: 生成文档ID
                # 为文档生成唯一的UUID
                doc_id = get_uuid()

                # 步骤3.10: 生成缩略图
                # 根据文件类型生成缩略图（如PDF首页、图片缩略图等）
                img = thumbnail_img(filename, blob)
                thumbnail_location = ""
                if img is not None:
                    # 如果成功生成缩略图，存储到对象存储
                    thumbnail_location = f"thumbnail_{doc_id}.png"
                    STORAGE_IMPL.put(kb.id, thumbnail_location, img)

                # 步骤3.11: 构建文档元数据
                # 创建包含文档所有信息的字典
                doc = {
                    "id": doc_id,                    # 文档唯一ID
                    "kb_id": kb.id,                  # 知识库ID
                    "parser_id": self.get_parser(filetype, filename, kb.parser_id),  # 解析器类型
                    "parser_config": kb.parser_config,  # 解析器配置
                    "created_by": user_id,           # 创建者ID
                    "type": filetype,                # 文件类型
                    "name": filename,                # 文件名
                    "suffix": Path(filename).suffix.lstrip("."),  # 文件扩展名
                    "location": location,            # 存储位置
                    "size": len(blob),              # 文件大小（字节）
                    "thumbnail": thumbnail_location, # 缩略图位置
                }
                
                # 步骤3.12: 保存文档元数据到数据库
                # 将文档信息插入到数据库的Document表中
                DocumentService.insert(doc)

                # 步骤3.13: 建立文件与文件夹的关联
                # 将文档添加到知识库文件夹中，建立文件系统结构
                FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
                
                # 步骤3.14: 添加到成功列表
                # 将文档信息和文件内容添加到成功处理列表
                files.append((doc, blob))
                
            except Exception as e:
                # 步骤3.15: 错误处理
                # 如果处理过程中出现异常，记录错误信息
                err.append(file.filename + ": " + str(e))

        # 步骤4: 返回处理结果
        # 返回错误列表和成功处理的文件列表
        return err, files

    @staticmethod
    def parse_docs(file_objs, user_id):
        from rag.app import audio, email, naive, picture, presentation

        def dummy(prog=None, msg=""):
            pass

        FACTORY = {ParserType.PRESENTATION.value: presentation, ParserType.PICTURE.value: picture, ParserType.AUDIO.value: audio, ParserType.EMAIL.value: email}
        parser_config = {"chunk_token_num": 16096, "delimiter": "\n!?;。；！？", "layout_recognize": "Plain Text"}
        exe = ThreadPoolExecutor(max_workers=12)
        threads = []
        for file in file_objs:
            kwargs = {"lang": "English", "callback": dummy, "parser_config": parser_config, "from_page": 0, "to_page": 100000, "tenant_id": user_id}
            filetype = filename_type(file.filename)
            blob = file.read()
            threads.append(exe.submit(FACTORY.get(FileService.get_parser(filetype, file.filename, ""), naive).chunk, file.filename, blob, **kwargs))

        res = []
        for th in threads:
            res.append("\n".join([ck["content_with_weight"] for ck in th.result()]))

        return "\n\n".join(res)

    @staticmethod
    def get_parser(doc_type, filename, default):
        if doc_type == FileType.VISUAL:
            return ParserType.PICTURE.value
        if doc_type == FileType.AURAL:
            return ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION.value
        if re.search(r"\.(eml)$", filename):
            return ParserType.EMAIL.value
        return default
