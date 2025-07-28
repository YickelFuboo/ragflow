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
import json
import logging
import re
import secrets
from datetime import datetime

from flask import redirect, request, session
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from api import settings
from api.apps.auth import get_auth_client
from api.db import FileType, UserTenantRole
from api.db.db_models import TenantLLM
from api.db.services.file_service import FileService
from api.db.services.llm_service import LLMService, TenantLLMService
from api.db.services.user_service import TenantService, UserService, UserTenantService
from api.utils import (
    current_timestamp,
    datetime_format,
    decrypt,
    download_img,
    get_format_time,
    get_uuid,
)
from api.utils.api_utils import (
    construct_response,
    get_data_error_result,
    get_json_result,
    server_error_response,
    validate_request,
)

@manager.route("/login", methods=["POST", "GET"])  # noqa: F821
def login():
    """
    用户登录接口
    
    功能：
    - 验证用户邮箱和密码
    - 生成新的访问令牌
    - 更新用户登录时间和状态
    - 执行用户登录会话
    
    请求参数：
    - email: 用户邮箱地址
    - password: 用户密码（加密后的）
    
    返回：
    - 成功：返回用户信息和认证令牌
    - 失败：返回错误信息
    """
    # 检查请求是否包含JSON数据
    if not request.json:
        return get_json_result(data=False, code=settings.RetCode.AUTHENTICATION_ERROR, message="Unauthorized!")

    # 获取用户邮箱
    email = request.json.get("email", "")
    # 根据邮箱查询用户是否存在
    users = UserService.query(email=email)
    if not users:
        return get_json_result(
            data=False,
            code=settings.RetCode.AUTHENTICATION_ERROR,
            message=f"Email: {email} is not registered!",
        )

    # 获取并解密密码
    password = request.json.get("password")
    try:
        password = decrypt(password)
    except BaseException:
        return get_json_result(data=False, code=settings.RetCode.SERVER_ERROR, message="Fail to crypt password")

    # 验证用户邮箱和密码是否匹配
    user = UserService.query_user(email, password)
    if user:
        # 登录成功，准备响应数据
        response_data = user.to_json()
        # 生成新的访问令牌
        user.access_token = get_uuid()
        # 执行Flask-Login登录：
        #   生成会话ID：Flask-Login 生成唯一的会话标识符
        #   存储用户信息：将会话ID和用户ID关联存储在服务器端
        #   设置Cookie：在客户端设置包含会话ID的Cookie
        #   更新用户状态：标记用户为已登录状态
        login_user(user)
        # 更新用户最后登录时间和日期
        user.update_time = (current_timestamp(),)
        user.update_date = (datetime_format(datetime.now()),)
        # 保存用户信息到数据库
        user.save()
        msg = "Welcome back!"
        # 返回成功响应，包含用户信息和认证令牌
        return construct_response(data=response_data, auth=user.get_id(), message=msg)
    else:
        # 密码验证失败
        return get_json_result(
            data=False,
            code=settings.RetCode.AUTHENTICATION_ERROR,
            message="Email and password do not match!",
        )


@manager.route("/login/channels", methods=["GET"])  # noqa: F821
def get_login_channels():
    """
    获取所有支持的认证渠道
    
    功能：
    - 从配置文件中读取OAuth配置
    - 返回所有可用的登录渠道信息
    - 包含渠道名称、显示名称和图标
    
    返回：
    - 成功：返回渠道列表
    - 失败：返回空列表和错误信息
    """
    try:
        channels = []
        # 遍历OAuth配置中的所有渠道
        for channel, config in settings.OAUTH_CONFIG.items():
            channels.append(
                {
                    "channel": channel,  # 渠道标识符
                    "display_name": config.get("display_name", channel.title()),  # 显示名称
                    "icon": config.get("icon", "sso"),  # 渠道图标
                }
            )
        return get_json_result(data=channels)
    except Exception as e:
        logging.exception(e)
        return get_json_result(data=[], message=f"Load channels failure, error: {str(e)}", code=settings.RetCode.EXCEPTION_ERROR)


@manager.route("/login/<channel>", methods=["GET"])  # noqa: F821
def oauth_login(channel):
    """
    OAuth登录重定向接口
    
    功能：
    - 根据指定的渠道名称获取OAuth配置
    - 生成OAuth状态令牌防止CSRF攻击
    - 重定向到第三方认证服务商的授权页面
    
    参数：
    - channel: OAuth渠道名称（如github、feishu等）
    
    返回：
    - 重定向到第三方认证页面
    - 失败时抛出异常
    """
    # 从配置中获取指定渠道的OAuth配置
    channel_config = settings.OAUTH_CONFIG.get(channel)
    if not channel_config:
        raise ValueError(f"Invalid channel name: {channel}")
    
    # 创建认证客户端
    auth_cli = get_auth_client(channel_config)

    # 生成OAuth状态令牌，用于防止CSRF攻击
    state = get_uuid()
    session["oauth_state"] = state
    
    # 获取授权URL并重定向
    auth_url = auth_cli.get_authorization_url(state)
    return redirect(auth_url)


@manager.route("/oauth/callback/<channel>", methods=["GET"])  # noqa: F821
def oauth_callback(channel):
    """
    OAuth回调处理接口
    
    功能：
    - 处理OAuth/OIDC认证回调
    - 验证状态令牌防止CSRF攻击
    - 交换授权码获取访问令牌
    - 获取用户信息并自动注册或登录
    - 支持多种OAuth渠道的动态处理
    
    参数：
    - channel: OAuth渠道名称
    
    查询参数：
    - state: OAuth状态令牌
    - code: 授权码
    
    返回：
    - 成功：重定向到前端并携带认证令牌
    - 失败：重定向到前端并携带错误信息
    """
    try:
        # 获取渠道配置并创建认证客户端
        channel_config = settings.OAUTH_CONFIG.get(channel)
        if not channel_config:
            raise ValueError(f"Invalid channel name: {channel}")
        auth_cli = get_auth_client(channel_config)

        # 验证OAuth状态令牌，防止CSRF攻击
        state = request.args.get("state")
        if not state or state != session.get("oauth_state"):
            return redirect("/?error=invalid_state")
        session.pop("oauth_state", None)  # 清除已使用的状态令牌

        # 获取授权码
        code = request.args.get("code")
        if not code:
            return redirect("/?error=missing_code")

        # 使用授权码交换访问令牌
        token_info = auth_cli.exchange_code_for_token(code)
        access_token = token_info.get("access_token")
        if not access_token:
            return redirect("/?error=token_failed")

        # 获取ID令牌（如果支持）
        id_token = token_info.get("id_token")

        # 使用访问令牌获取用户信息
        user_info = auth_cli.fetch_user_info(access_token, id_token=id_token)
        if not user_info.email:
            return redirect("/?error=email_missing")

        # 检查用户是否已存在
        users = UserService.query(email=user_info.email)
        user_id = get_uuid()

        if not users:
            # 用户不存在，执行自动注册
            try:
                # 尝试下载用户头像
                try:
                    avatar = download_img(user_info.avatar_url)
                except Exception as e:
                    logging.exception(e)
                    avatar = ""

                # 注册新用户
                users = user_register(
                    user_id,
                    {
                        "access_token": get_uuid(),
                        "email": user_info.email,
                        "avatar": avatar,
                        "nickname": user_info.nickname,
                        "login_channel": channel,
                        "last_login_time": get_format_time(),
                        "is_superuser": False,
                    },
                )

                if not users:
                    raise Exception(f"Failed to register {user_info.email}")
                if len(users) > 1:
                    raise Exception(f"Same email: {user_info.email} exists!")

                # 注册成功后执行登录
                user = users[0]
                login_user(user)
                return redirect(f"/?auth={user.get_id()}")

            except Exception as e:
                # 注册失败，回滚操作
                rollback_user_registration(user_id)
                logging.exception(e)
                return redirect(f"/?error={str(e)}")

        # 用户已存在，执行登录
        user = users[0]
        user.access_token = get_uuid()
        login_user(user)
        user.save()
        return redirect(f"/?auth={user.get_id()}")
    except Exception as e:
        logging.exception(e)
        return redirect(f"/?error={str(e)}")


@manager.route("/github_callback", methods=["GET"])  # noqa: F821
def github_callback():
    """
    GitHub OAuth回调接口（已弃用）
    
    注意：此接口已弃用，请使用 `/oauth/callback/<channel>` 接口
    
    功能：
    - 处理GitHub OAuth认证回调
    - 交换授权码获取访问令牌
    - 获取GitHub用户信息和邮箱
    - 自动注册或登录用户
    
    查询参数：
    - code: GitHub返回的授权码
    
    返回：
    - 成功：重定向到前端并携带认证令牌
    - 失败：重定向到前端并携带错误信息
    """
    import requests

    # 使用授权码向GitHub交换访问令牌
    res = requests.post(
        settings.GITHUB_OAUTH.get("url"),
        data={
            "client_id": settings.GITHUB_OAUTH.get("client_id"),
            "client_secret": settings.GITHUB_OAUTH.get("secret_key"),
            "code": request.args.get("code"),
        },
        headers={"Accept": "application/json"},
    )
    res = res.json()
    
    # 检查GitHub返回的错误
    if "error" in res:
        return redirect("/?error=%s" % res["error_description"])

    # 验证授权范围是否包含邮箱权限
    if "user:email" not in res["scope"].split(","):
        return redirect("/?error=user:email not in scope")

    # 保存访问令牌到会话
    session["access_token"] = res["access_token"]
    session["access_token_from"] = "github"
    
    # 获取GitHub用户信息
    user_info = user_info_from_github(session["access_token"])
    email_address = user_info["email"]
    
    # 检查用户是否已存在
    users = UserService.query(email=email_address)
    user_id = get_uuid()
    if not users:
        # 用户不存在，执行自动注册
        try:
            # 尝试下载用户头像
            try:
                avatar = download_img(user_info["avatar_url"])
            except Exception as e:
                logging.exception(e)
                avatar = ""
            
            # 注册新用户
            users = user_register(
                user_id,
                {
                    "access_token": session["access_token"],
                    "email": email_address,
                    "avatar": avatar,
                    "nickname": user_info["login"],
                    "login_channel": "github",
                    "last_login_time": get_format_time(),
                    "is_superuser": False,
                },
            )
            if not users:
                raise Exception(f"Fail to register {email_address}.")
            if len(users) > 1:
                raise Exception(f"Same email: {email_address} exists!")

            # 注册成功后执行登录
            user = users[0]
            login_user(user)
            return redirect("/?auth=%s" % user.get_id())
        except Exception as e:
            # 注册失败，回滚操作
            rollback_user_registration(user_id)
            logging.exception(e)
            return redirect("/?error=%s" % str(e))

    # 用户已存在，执行登录
    user = users[0]
    user.access_token = get_uuid()
    login_user(user)
    user.save()
    return redirect("/?auth=%s" % user.get_id())


@manager.route("/feishu_callback", methods=["GET"])  # noqa: F821
def feishu_callback():
    """
    Feishu OAuth callback endpoint.
    ---
    tags:
      - OAuth
    parameters:
      - in: query
        name: code
        type: string
        required: true
        description: Authorization code from Feishu.
    responses:
      200:
        description: Authentication successful.
        schema:
          type: object
    """
    import requests

    app_access_token_res = requests.post(
        settings.FEISHU_OAUTH.get("app_access_token_url"),
        data=json.dumps(
            {
                "app_id": settings.FEISHU_OAUTH.get("app_id"),
                "app_secret": settings.FEISHU_OAUTH.get("app_secret"),
            }
        ),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    app_access_token_res = app_access_token_res.json()
    if app_access_token_res["code"] != 0:
        return redirect("/?error=%s" % app_access_token_res)

    res = requests.post(
        settings.FEISHU_OAUTH.get("user_access_token_url"),
        data=json.dumps(
            {
                "grant_type": settings.FEISHU_OAUTH.get("grant_type"),
                "code": request.args.get("code"),
            }
        ),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {app_access_token_res['app_access_token']}",
        },
    )
    res = res.json()
    if res["code"] != 0:
        return redirect("/?error=%s" % res["message"])

    if "contact:user.email:readonly" not in res["data"]["scope"].split():
        return redirect("/?error=contact:user.email:readonly not in scope")
    session["access_token"] = res["data"]["access_token"]
    session["access_token_from"] = "feishu"
    user_info = user_info_from_feishu(session["access_token"])
    email_address = user_info["email"]
    users = UserService.query(email=email_address)
    user_id = get_uuid()
    if not users:
        # User isn't try to register
        try:
            try:
                avatar = download_img(user_info["avatar_url"])
            except Exception as e:
                logging.exception(e)
                avatar = ""
            users = user_register(
                user_id,
                {
                    "access_token": session["access_token"],
                    "email": email_address,
                    "avatar": avatar,
                    "nickname": user_info["en_name"],
                    "login_channel": "feishu",
                    "last_login_time": get_format_time(),
                    "is_superuser": False,
                },
            )
            if not users:
                raise Exception(f"Fail to register {email_address}.")
            if len(users) > 1:
                raise Exception(f"Same email: {email_address} exists!")

            # Try to log in
            user = users[0]
            login_user(user)
            return redirect("/?auth=%s" % user.get_id())
        except Exception as e:
            rollback_user_registration(user_id)
            logging.exception(e)
            return redirect("/?error=%s" % str(e))

    # User has already registered, try to log in
    user = users[0]
    user.access_token = get_uuid()
    login_user(user)
    user.save()
    return redirect("/?auth=%s" % user.get_id())


def user_info_from_feishu(access_token):
    import requests

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {access_token}",
    }
    res = requests.get("https://open.feishu.cn/open-apis/authen/v1/user_info", headers=headers)
    user_info = res.json()["data"]
    user_info["email"] = None if user_info.get("email") == "" else user_info["email"]
    return user_info


def user_info_from_github(access_token):
    import requests

    headers = {"Accept": "application/json", "Authorization": f"token {access_token}"}
    res = requests.get(f"https://api.github.com/user?access_token={access_token}", headers=headers)
    user_info = res.json()
    email_info = requests.get(
        f"https://api.github.com/user/emails?access_token={access_token}",
        headers=headers,
    ).json()
    user_info["email"] = next((email for email in email_info if email["primary"]), None)["email"]
    return user_info


@manager.route("/logout", methods=["GET"])  # noqa: F821
@login_required
def log_out():
    """
    用户登出接口
    
    功能：
    - 使当前用户的访问令牌失效
    - 清除Flask-Login会话
    - 确保用户安全退出系统
    
    安全机制：
    - 将访问令牌标记为无效，防止令牌被重用
    - 清除服务器端会话信息
    - 要求用户必须已登录才能执行登出操作
    
    返回：
    - 成功：返回登出成功状态
    - 失败：如果用户未登录，会重定向到登录页面
    """
    # 生成无效的访问令牌，使当前令牌失效
    # 格式：INVALID_ + 32位随机十六进制字符串
    current_user.access_token = f"INVALID_{secrets.token_hex(16)}"
    
    # 将失效的令牌保存到数据库
    current_user.save()
    
    # 清除Flask-Login会话，注销用户
    logout_user()
    
    # 返回登出成功响应
    return get_json_result(data=True)


@manager.route("/setting", methods=["POST"])  # noqa: F821
@login_required
def setting_user():
    """
    更新用户设置接口
    
    功能：
    - 允许用户更新个人信息（昵称、头像等）
    - 支持密码修改（需要验证原密码）
    - 保护敏感字段不被随意修改
    
    安全限制：
    - 只允许修改非敏感字段
    - 密码修改需要验证原密码
    - 禁止修改邮箱、状态、权限等关键字段
    
    请求参数：
    - nickname: 新昵称
    - avatar: 新头像
    - password: 原密码（修改密码时需要）
    - new_password: 新密码（修改密码时需要）
    - 其他非敏感字段
    
    返回：
    - 成功：返回更新成功状态
    - 失败：返回错误信息和错误代码
    """
    update_dict = {}  # 存储需要更新的字段
    request_data = request.json
    
    # 处理密码修改逻辑
    if request_data.get("password"):
        new_password = request_data.get("new_password")
        
        # 验证原密码是否正确
        if not check_password_hash(current_user.password, decrypt(request_data["password"])):
            return get_json_result(
                data=False,
                code=settings.RetCode.AUTHENTICATION_ERROR,
                message="Password error!",
            )

        # 如果提供了新密码，则更新密码
        if new_password:
            update_dict["password"] = generate_password_hash(decrypt(new_password))

    # 过滤允许更新的字段，保护敏感字段
    for k in request_data.keys():
        # 以下字段不允许通过此接口修改，需要特殊权限
        if k in [
            "password",           # 密码通过上面的逻辑处理
            "new_password",       # 新密码通过上面的逻辑处理
            "email",             # 邮箱不允许修改
            "status",            # 用户状态不允许修改
            "is_superuser",      # 超级用户权限不允许修改
            "login_channel",     # 登录渠道不允许修改
            "is_anonymous",      # 匿名状态不允许修改
            "is_active",         # 激活状态不允许修改
            "is_authenticated",  # 认证状态不允许修改
            "last_login_time",   # 最后登录时间不允许修改
        ]:
            continue
        
        # 将允许修改的字段添加到更新字典中
        update_dict[k] = request_data[k]

    try:
        # 调用服务层更新用户信息
        UserService.update_by_id(current_user.id, update_dict)
        return get_json_result(data=True)
    except Exception as e:
        # 记录异常日志
        logging.exception(e)
        return get_json_result(data=False, message="Update failure!", code=settings.RetCode.EXCEPTION_ERROR)


@manager.route("/info", methods=["GET"])  # noqa: F821
@login_required
def user_profile():
    """
    获取用户个人信息接口
    
    功能：
    - 返回当前登录用户的完整个人信息
    - 包括用户ID、昵称、邮箱、头像等基本信息
    - 用于前端显示用户资料和设置页面
    
    安全机制：
    - 要求用户必须已登录才能获取个人信息
    - 只能获取当前登录用户的信息，不能获取其他用户信息
    - 通过Flask-Login的@login_required装饰器保护
    
    返回数据：
    - id: 用户唯一标识符
    - nickname: 用户昵称
    - email: 用户邮箱地址
    - avatar: 用户头像（base64编码）
    - language: 用户语言偏好
    - color_schema: 用户颜色主题偏好
    - timezone: 用户时区设置
    - 其他用户相关字段
    
    使用场景：
    - 用户个人资料页面
    - 用户设置页面
    - 导航栏用户信息显示
    """
    # 直接返回当前用户的完整信息字典
    # current_user 是 Flask-Login 提供的当前登录用户对象
    # to_dict() 方法将用户对象转换为字典格式
    return get_json_result(data=current_user.to_dict())


def rollback_user_registration(user_id):
    try:
        UserService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        TenantService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        u = UserTenantService.query(tenant_id=user_id)
        if u:
            UserTenantService.delete_by_id(u[0].id)
    except Exception:
        pass
    try:
        TenantLLM.delete().where(TenantLLM.tenant_id == user_id).execute()
    except Exception:
        pass


def user_register(user_id, user):
    """
    用户注册函数
    
    功能：
    - 创建新用户记录并保存到数据库
    - 为新用户创建默认的租户信息
    - 建立用户与租户的关联关系
    - 设置用户的默认LLM配置和文件系统
    
    参数：
    - user_id: 用户唯一标识符
    - user: 包含用户信息的字典
    
    用户信息字典包含：
    - access_token: 用户访问令牌
    - email: 用户邮箱地址
    - avatar: 用户头像
    - nickname: 用户昵称
    - login_channel: 登录渠道
    - last_login_time: 最后登录时间
    - is_superuser: 是否为超级用户
    
    返回：
    - 成功：返回用户对象列表
    - 失败：返回空列表或抛出异常
    
    数据库操作：
    - 创建用户记录
    - 创建租户记录
    - 建立用户-租户关联
    - 设置默认LLM配置
    - 创建用户文件系统根目录
    """
    # 设置用户ID
    user["id"] = user_id
    
    # 创建用户默认租户信息
    tenant = {
        "id": user_id,
        "name": user["nickname"] + "'s Kingdom",  # 租户名称：用户昵称 + 's Kingdom
        "llm_id": settings.CHAT_MDL,  # 默认聊天模型
        "embd_id": settings.EMBEDDING_MDL,  # 默认嵌入模型
        "asr_id": settings.ASR_MDL,  # 默认语音识别模型
        "parser_ids": settings.PARSERS,  # 默认解析器
        "img2txt_id": settings.IMAGE2TEXT_MDL,  # 默认图像转文本模型
        "rerank_id": settings.RERANK_MDL,  # 默认重排序模型
    }
    
    # 创建用户-租户关联信息
    usr_tenant = {
        "tenant_id": user_id,
        "user_id": user_id,
        "invited_by": user_id,  # 邀请人（自己）
        "role": UserTenantRole.OWNER,  # 角色：所有者
    }
    
    # 创建用户文件系统根目录
    file_id = get_uuid()
    file = {
        "id": file_id,
        "parent_id": file_id,  # 根目录的父目录就是自己
        "tenant_id": user_id,
        "created_by": user_id,
        "name": "/",
        "type": FileType.FOLDER.value,
        "size": 0,
        "location": "",
    }
    # 初始化租户LLM配置列表
    tenant_llm = []
    
    # 查询LLM工厂中的所有LLM模型，并为每个模型创建租户配置
    for llm in LLMService.query(fid=settings.LLM_FACTORY):
        tenant_llm.append(
            {
                "tenant_id": user_id,  # 租户ID
                "llm_factory": settings.LLM_FACTORY,  # LLM工厂标识
                "llm_name": llm.llm_name,  # LLM模型名称
                "model_type": llm.model_type,  # 模型类型
                "api_key": settings.API_KEY,  # API密钥
                "api_base": settings.LLM_BASE_URL,  # API基础URL
                "max_tokens": llm.max_tokens if llm.max_tokens else 8192,  # 最大token数，默认8192
            }
        )
    
    # 如果不是轻量模式，添加内置嵌入模型配置
    if settings.LIGHTEN != 1:
        # 遍历所有内置嵌入模型
        for buildin_embedding_model in settings.BUILTIN_EMBEDDING_MODELS:
            # 分离模型名称和工厂标识
            mdlnm, fid = TenantLLMService.split_model_name_and_factory(buildin_embedding_model)
            tenant_llm.append(
                {
                    "tenant_id": user_id,  # 租户ID
                    "llm_factory": fid,  # 嵌入模型工厂标识
                    "llm_name": mdlnm,  # 嵌入模型名称
                    "model_type": "embedding",  # 模型类型为嵌入
                    "api_key": "",  # 嵌入模型不需要API密钥
                    "api_base": "",  # 嵌入模型不需要API基础URL
                    # 根据模型类型设置最大token数
                    "max_tokens": 1024 if buildin_embedding_model == "BAAI/bge-large-zh-v1.5@BAAI" else 512,
                }
            )

    # 保存用户信息，如果保存失败则返回
    if not UserService.save(**user):
        return
    
    # 插入租户信息
    TenantService.insert(**tenant)
    
    # 插入用户租户关联信息
    UserTenantService.insert(**usr_tenant)
    
    # 批量插入租户LLM配置
    TenantLLMService.insert_many(tenant_llm)
    
    # 插入文件信息
    FileService.insert(file)
    
    # 返回查询到的用户信息
    return UserService.query(email=user["email"])


@manager.route("/register", methods=["POST"])  # noqa: F821
@validate_request("nickname", "email", "password")
def user_add():
    """
    用户注册接口
    
    功能：
    - 允许新用户通过邮箱和密码注册账户
    - 验证邮箱格式和唯一性
    - 创建用户账户并设置默认配置
    - 支持注册功能的开关控制
    
    安全验证：
    - 验证请求参数完整性（昵称、邮箱、密码）
    - 验证邮箱格式的有效性
    - 检查邮箱是否已被注册
    - 密码加密存储
    
    请求参数：
    - nickname: 用户昵称
    - email: 用户邮箱地址
    - password: 用户密码（加密后的）
    
    返回：
    - 成功：返回注册成功状态
    - 失败：返回错误信息和错误代码
    
    使用场景：
    - 新用户注册页面
    - 用户自主创建账户
    - 系统用户管理
    """
    
    # 检查注册功能是否启用
    if not settings.REGISTER_ENABLED:
        return get_json_result(
            data=False,
            message="User registration is disabled!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    req = request.json
    email_address = req["email"]

    # 验证邮箱地址格式
    if not re.match(r"^[\w\._-]+@([\w_-]+\.)+[\w-]{2,}$", email_address):
        return get_json_result(
            data=False,
            message=f"Invalid email address: {email_address}!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    # 检查邮箱地址是否已被注册
    if UserService.query(email=email_address):
        return get_json_result(
            data=False,
            message=f"Email: {email_address} has already registered!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    # 构建用户信息数据
    nickname = req["nickname"]
    user_dict = {
        "access_token": get_uuid(),  # 生成访问令牌
        "email": email_address,  # 用户邮箱
        "nickname": nickname,  # 用户昵称
        "password": decrypt(req["password"]),  # 解密并存储密码
        "login_channel": "password",  # 登录渠道：密码登录
        "last_login_time": get_format_time(),  # 最后登录时间
        "is_superuser": False,  # 非超级用户
    }

    # 生成唯一的用户ID
    user_id = get_uuid()
    
    try:
        # 调用用户注册函数，创建用户记录和相关配置
        users = user_register(user_id, user_dict)
        
        # 检查注册是否成功，如果没有返回用户信息则注册失败
        if not users:
            raise Exception(f"Fail to register {email_address}.")
        
        # 检查是否存在重复邮箱，如果返回多个用户说明邮箱已存在
        if len(users) > 1:
            raise Exception(f"Same email: {email_address} exists!")
        
        # 获取注册成功的用户信息
        user = users[0]
        
        # 使用Flask-Login登录用户，建立会话
        login_user(user)
        
        # 返回注册成功的响应，包含用户信息和认证令牌
        return construct_response(
            data=user.to_json(),  # 用户数据
            auth=user.get_id(),   # 认证令牌
            message=f"{nickname}, welcome aboard!",  # 欢迎消息
        )
        
    except Exception as e:
        # 如果注册过程中出现异常，回滚用户注册操作
        rollback_user_registration(user_id)
        
        # 记录异常日志
        logging.exception(e)
        
        # 返回注册失败的响应
        return get_json_result(
            data=False,
            message=f"User registration failure, error: {str(e)}",
            code=settings.RetCode.EXCEPTION_ERROR,
        )


@manager.route("/tenant_info", methods=["GET"])  # noqa: F821
@login_required
def tenant_info():
    """
    获取租户信息接口
    
    功能：
    - 获取当前登录用户所属的租户信息
    - 返回租户的配置信息（LLM、嵌入模型等）
    - 用于前端显示租户设置和配置
    
    安全机制：
    - 要求用户必须已登录才能获取租户信息
    - 只能获取当前用户所属租户的信息
    - 通过Flask-Login的@login_required装饰器保护
    
    返回数据：
    - tenant_id: 租户ID
    - tenant_name: 租户名称
    - llm_id: 聊天模型ID
    - embd_id: 嵌入模型ID
    - asr_id: 语音识别模型ID
    - img2txt_id: 图像转文本模型ID
    - parser_ids: 解析器ID列表
    - rerank_id: 重排序模型ID
    
    使用场景：
    - 租户设置页面
    - 模型配置管理
    - 系统配置显示
    """
    # 获取当前用户的租户信息
    tenant = TenantService.query(id=current_user.id)
    if not tenant:
        return get_json_result(data=False, message="Tenant not found!", code=settings.RetCode.OPERATING_ERROR)
    
    # 返回租户信息
    return get_json_result(data=tenant[0].to_dict())


@manager.route("/set_tenant_info", methods=["POST"])  # noqa: F821
@login_required
@validate_request("tenant_id", "asr_id", "embd_id", "img2txt_id", "llm_id")
def set_tenant_info():
    """
    设置租户信息接口
    
    功能：
    - 更新当前用户所属租户的配置信息
    - 支持修改各种AI模型的配置
    - 验证用户权限并更新租户设置
    
    安全机制：
    - 要求用户必须已登录才能修改租户信息
    - 验证请求参数完整性
    - 只能修改当前用户所属租户的信息
    - 通过Flask-Login的@login_required装饰器保护
    
    请求参数：
    - tenant_id: 租户ID
    - asr_id: 语音识别模型ID
    - embd_id: 嵌入模型ID
    - img2txt_id: 图像转文本模型ID
    - llm_id: 聊天模型ID
    
    返回：
    - 成功：返回更新成功状态
    - 失败：返回错误信息和错误代码
    
    使用场景：
    - 租户设置页面
    - 模型配置管理
    - 系统参数调整
    """
    try:
        # 获取请求数据
        req = request.json
        
        # 验证用户是否有权限修改该租户
        tenant = TenantService.query(id=req["tenant_id"])
        if not tenant:
            return get_json_result(
                data=False, 
                message="Tenant not found!", 
                code=settings.RetCode.OPERATING_ERROR
            )
        
        # 更新租户配置信息
        tenant[0].asr_id = req["asr_id"]  # 更新语音识别模型
        tenant[0].embd_id = req["embd_id"]  # 更新嵌入模型
        tenant[0].img2txt_id = req["img2txt_id"]  # 更新图像转文本模型
        tenant[0].llm_id = req["llm_id"]  # 更新聊天模型
        
        # 保存更新后的租户信息
        tenant[0].save()
        
        return get_json_result(data=True, message="Tenant info updated successfully!")
        
    except Exception as e:
        # 记录异常日志
        logging.exception(e)
        return get_json_result(
            data=False, 
            message="Update failure!", 
            code=settings.RetCode.EXCEPTION_ERROR
        )
