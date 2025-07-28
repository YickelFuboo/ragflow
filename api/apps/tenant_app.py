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

from flask import request
from flask_login import login_required, current_user

from api import settings
from api.db import UserTenantRole, StatusEnum
from api.db.db_models import UserTenant
from api.db.services.user_service import UserTenantService, UserService

from api.utils import get_uuid, delta_seconds
from api.utils.api_utils import get_json_result, validate_request, server_error_response, get_data_error_result


@manager.route("/<tenant_id>/user/list", methods=["GET"])  # noqa: F821
@login_required
def user_list(tenant_id):
    """
    获取租户用户列表接口
    
    功能：获取指定租户下的所有用户列表
    权限：只有租户所有者才能查看用户列表
    """
    # 验证当前用户是否为租户所有者
    if current_user.id != tenant_id:
        return get_json_result(
            data=False,
            message='No authorization.',
            code=settings.RetCode.AUTHENTICATION_ERROR)

    try:
        # 查询租户下的所有用户
        users = UserTenantService.get_by_tenant_id(tenant_id)
        # 为每个用户添加时间差信息
        for u in users:
            u["delta_seconds"] = delta_seconds(str(u["update_date"]))
        return get_json_result(data=users)
    except Exception as e:
        return server_error_response(e)


@manager.route('/<tenant_id>/user', methods=['POST'])  # noqa: F821
@login_required
@validate_request("email")
def create(tenant_id):
    """
    邀请用户加入租户接口
    
    功能：邀请指定邮箱的用户加入租户
    权限：只有租户所有者才能邀请用户
    参数：email - 被邀请用户的邮箱地址
    """
    # 验证当前用户是否为租户所有者
    if current_user.id != tenant_id:
        return get_json_result(
            data=False,
            message='No authorization.',
            code=settings.RetCode.AUTHENTICATION_ERROR)

    req = request.json
    invite_user_email = req["email"]
    
    # 查询被邀请用户是否存在
    invite_users = UserService.query(email=invite_user_email)
    if not invite_users:
        return get_data_error_result(message="User not found.")

    user_id_to_invite = invite_users[0].id
    
    # 检查用户是否已经在租户中
    user_tenants = UserTenantService.query(user_id=user_id_to_invite, tenant_id=tenant_id)
    if user_tenants:
        user_tenant_role = user_tenants[0].role
        # 检查用户角色状态
        if user_tenant_role == UserTenantRole.NORMAL:
            return get_data_error_result(message=f"{invite_user_email} is already in the team.")
        if user_tenant_role == UserTenantRole.OWNER:
            return get_data_error_result(message=f"{invite_user_email} is the owner of the team.")
        return get_data_error_result(message=f"{invite_user_email} is in the team, but the role: {user_tenant_role} is invalid.")

    # 创建用户租户关联记录，设置角色为邀请状态
    UserTenantService.save(
        id=get_uuid(),
        user_id=user_id_to_invite,
        tenant_id=tenant_id,
        invited_by=current_user.id,  # 邀请人
        role=UserTenantRole.INVITE,  # 角色：邀请中
        status=StatusEnum.VALID.value)

    # 返回被邀请用户的基本信息
    usr = invite_users[0].to_dict()
    usr = {k: v for k, v in usr.items() if k in ["id", "avatar", "email", "nickname"]}

    return get_json_result(data=usr)


@manager.route('/<tenant_id>/user/<user_id>', methods=['DELETE'])  # noqa: F821
@login_required
def rm(tenant_id, user_id):
    """
    移除租户用户接口
    
    功能：从租户中移除指定用户
    权限：租户所有者可以移除任何用户，用户也可以主动退出
    """
    # 验证权限：只有租户所有者或用户本人可以执行此操作
    if current_user.id != tenant_id and current_user.id != user_id:
        return get_json_result(
            data=False,
            message='No authorization.',
            code=settings.RetCode.AUTHENTICATION_ERROR)

    try:
        # 删除用户租户关联记录
        UserTenantService.filter_delete([UserTenant.tenant_id == tenant_id, UserTenant.user_id == user_id])
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/list", methods=["GET"])  # noqa: F821
@login_required
def tenant_list():
    """
    获取用户加入的租户列表接口
    
    功能：获取当前用户加入的所有租户列表
    """
    try:
        # 查询用户加入的所有租户
        users = UserTenantService.get_tenants_by_user_id(current_user.id)
        # 为每个租户添加时间差信息
        for u in users:
            u["delta_seconds"] = delta_seconds(str(u["update_date"]))
        return get_json_result(data=users)
    except Exception as e:
        return server_error_response(e)


@manager.route("/agree/<tenant_id>", methods=["PUT"])  # noqa: F821
@login_required
def agree(tenant_id):
    """
    同意租户邀请接口
    
    功能：用户同意加入指定租户
    权限：只有被邀请的用户可以执行此操作
    """
    try:
        # 更新用户租户关联记录，将角色从邀请状态改为普通成员
        UserTenantService.filter_update([UserTenant.tenant_id == tenant_id, UserTenant.user_id == current_user.id], {"role": UserTenantRole.NORMAL})
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
