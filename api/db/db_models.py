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
import hashlib
import inspect
import logging
import operator
import os
import sys
import time
import typing
from enum import Enum
from functools import wraps

from flask_login import UserMixin
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer
from peewee import BigIntegerField, BooleanField, CharField, CompositeKey, DateTimeField, Field, FloatField, IntegerField, Metadata, Model, TextField
from playhouse.migrate import MySQLMigrator, PostgresqlMigrator, migrate
from playhouse.pool import PooledMySQLDatabase, PooledPostgresqlDatabase

from api import settings, utils
from api.db import ParserType, SerializedType


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


CONTINUOUS_FIELD_TYPE = {IntegerField, FloatField, DateTimeField}
AUTO_DATE_TIMESTAMP_FIELD_PREFIX = {"create", "start", "end", "update", "read_access", "write_access"}


class TextFieldType(Enum):
    MYSQL = "LONGTEXT"
    POSTGRES = "TEXT"


class LongTextField(TextField):
    field_type = TextFieldType[settings.DATABASE_TYPE.upper()].value


class JSONField(LongTextField):
    default_value = {}

    def __init__(self, object_hook=None, object_pairs_hook=None, **kwargs):
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        super().__init__(**kwargs)

    def db_value(self, value):
        if value is None:
            value = self.default_value
        return utils.json_dumps(value)

    def python_value(self, value):
        if not value:
            return self.default_value
        return utils.json_loads(value, object_hook=self._object_hook, object_pairs_hook=self._object_pairs_hook)


class ListField(JSONField):
    default_value = []


class SerializedField(LongTextField):
    def __init__(self, serialized_type=SerializedType.PICKLE, object_hook=None, object_pairs_hook=None, **kwargs):
        self._serialized_type = serialized_type
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        super().__init__(**kwargs)

    def db_value(self, value):
        if self._serialized_type == SerializedType.PICKLE:
            return utils.serialize_b64(value, to_str=True)
        elif self._serialized_type == SerializedType.JSON:
            if value is None:
                return None
            return utils.json_dumps(value, with_type=True)
        else:
            raise ValueError(f"the serialized type {self._serialized_type} is not supported")

    def python_value(self, value):
        if self._serialized_type == SerializedType.PICKLE:
            return utils.deserialize_b64(value)
        elif self._serialized_type == SerializedType.JSON:
            if value is None:
                return {}
            return utils.json_loads(value, object_hook=self._object_hook, object_pairs_hook=self._object_pairs_hook)
        else:
            raise ValueError(f"the serialized type {self._serialized_type} is not supported")


def is_continuous_field(cls: typing.Type) -> bool:
    if cls in CONTINUOUS_FIELD_TYPE:
        return True
    for p in cls.__bases__:
        if p in CONTINUOUS_FIELD_TYPE:
            return True
        elif p is not Field and p is not object:
            if is_continuous_field(p):
                return True
    else:
        return False


def auto_date_timestamp_field():
    return {f"{f}_time" for f in AUTO_DATE_TIMESTAMP_FIELD_PREFIX}


def auto_date_timestamp_db_field():
    return {f"f_{f}_time" for f in AUTO_DATE_TIMESTAMP_FIELD_PREFIX}


def remove_field_name_prefix(field_name):
    return field_name[2:] if field_name.startswith("f_") else field_name


class BaseModel(Model):
    create_time = BigIntegerField(null=True, index=True)
    create_date = DateTimeField(null=True, index=True)
    update_time = BigIntegerField(null=True, index=True)
    update_date = DateTimeField(null=True, index=True)

    def to_json(self):
        # This function is obsolete
        return self.to_dict()

    def to_dict(self):
        return self.__dict__["__data__"]

    def to_human_model_dict(self, only_primary_with: list = None):
        model_dict = self.__dict__["__data__"]

        if not only_primary_with:
            return {remove_field_name_prefix(k): v for k, v in model_dict.items()}

        human_model_dict = {}
        for k in self._meta.primary_key.field_names:
            human_model_dict[remove_field_name_prefix(k)] = model_dict[k]
        for k in only_primary_with:
            human_model_dict[k] = model_dict[f"f_{k}"]
        return human_model_dict

    @property
    def meta(self) -> Metadata:
        return self._meta

    @classmethod
    def get_primary_keys_name(cls):
        return cls._meta.primary_key.field_names if isinstance(cls._meta.primary_key, CompositeKey) else [cls._meta.primary_key.name]

    @classmethod
    def getter_by(cls, attr):
        return operator.attrgetter(attr)(cls)

    @classmethod
    def query(cls, reverse=None, order_by=None, **kwargs):
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = "%s" % f_n
            if not hasattr(cls, attr_name) or f_v is None:
                continue
            if type(f_v) in {list, set}:
                f_v = list(f_v)
                if is_continuous_field(type(getattr(cls, attr_name))):
                    if len(f_v) == 2:
                        for i, v in enumerate(f_v):
                            if isinstance(v, str) and f_n in auto_date_timestamp_field():
                                # time type: %Y-%m-%d %H:%M:%S
                                f_v[i] = utils.date_string_to_timestamp(v)
                        lt_value = f_v[0]
                        gt_value = f_v[1]
                        if lt_value is not None and gt_value is not None:
                            filters.append(cls.getter_by(attr_name).between(lt_value, gt_value))
                        elif lt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) >= lt_value)
                        elif gt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) <= gt_value)
                else:
                    filters.append(operator.attrgetter(attr_name)(cls) << f_v)
            else:
                filters.append(operator.attrgetter(attr_name)(cls) == f_v)
        if filters:
            query_records = cls.select().where(*filters)
            if reverse is not None:
                if not order_by or not hasattr(cls, f"{order_by}"):
                    order_by = "create_time"
                if reverse is True:
                    query_records = query_records.order_by(cls.getter_by(f"{order_by}").desc())
                elif reverse is False:
                    query_records = query_records.order_by(cls.getter_by(f"{order_by}").asc())
            return [query_record for query_record in query_records]
        else:
            return []

    @classmethod
    def insert(cls, __data=None, **insert):
        if isinstance(__data, dict) and __data:
            __data[cls._meta.combined["create_time"]] = utils.current_timestamp()
        if insert:
            insert["create_time"] = utils.current_timestamp()

        return super().insert(__data, **insert)

    # update and insert will call this method
    @classmethod
    def _normalize_data(cls, data, kwargs):
        normalized = super()._normalize_data(data, kwargs)
        if not normalized:
            return {}

        normalized[cls._meta.combined["update_time"]] = utils.current_timestamp()

        for f_n in AUTO_DATE_TIMESTAMP_FIELD_PREFIX:
            if {f"{f_n}_time", f"{f_n}_date"}.issubset(cls._meta.combined.keys()) and cls._meta.combined[f"{f_n}_time"] in normalized and normalized[cls._meta.combined[f"{f_n}_time"]] is not None:
                normalized[cls._meta.combined[f"{f_n}_date"]] = utils.timestamp_to_date(normalized[cls._meta.combined[f"{f_n}_time"]])

        return normalized


class JsonSerializedField(SerializedField):
    def __init__(self, object_hook=utils.from_dict_hook, object_pairs_hook=None, **kwargs):
        super(JsonSerializedField, self).__init__(serialized_type=SerializedType.JSON, object_hook=object_hook, object_pairs_hook=object_pairs_hook, **kwargs)


class RetryingPooledMySQLDatabase(PooledMySQLDatabase):
    def __init__(self, *args, **kwargs):
        self.max_retries = kwargs.pop('max_retries', 5)
        self.retry_delay = kwargs.pop('retry_delay', 1)
        super().__init__(*args, **kwargs)

    def execute_sql(self, sql, params=None, commit=True):
        from peewee import OperationalError
        for attempt in range(self.max_retries + 1):
            try:
                return super().execute_sql(sql, params, commit)
            except OperationalError as e:
                if e.args[0] in (2013, 2006) and attempt < self.max_retries:
                    logging.warning(
                        f"Lost connection (attempt {attempt+1}/{self.max_retries}): {e}"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logging.error(f"DB execution failure: {e}")
                    raise
        return None

    def _handle_connection_loss(self):
        self.close_all()
        self.connect()

    def begin(self):
        from peewee import OperationalError
        for attempt in range(self.max_retries + 1):
            try:
                return super().begin()
            except OperationalError as e:
                if e.args[0] in (2013, 2006) and attempt < self.max_retries:
                    logging.warning(
                        f"Lost connection during transaction (attempt {attempt+1}/{self.max_retries})"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise


class PooledDatabase(Enum):
    MYSQL = RetryingPooledMySQLDatabase
    POSTGRES = PooledPostgresqlDatabase


class DatabaseMigrator(Enum):
    MYSQL = MySQLMigrator
    POSTGRES = PostgresqlMigrator


@singleton
class BaseDataBase:
    def __init__(self):
        database_config = settings.DATABASE.copy()
        db_name = database_config.pop("name")
        self.database_connection = PooledDatabase[settings.DATABASE_TYPE.upper()].value(db_name, **database_config)
        logging.info("init database on cluster mode successfully")


def with_retry(max_retries=3, retry_delay=1.0):
    """Decorator: Add retry mechanism to database operations

    Args:
        max_retries (int): maximum number of retries
        retry_delay (float): initial retry delay (seconds), will increase exponentially

    Returns:
        decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # get self and method name for logging
                    self_obj = args[0] if args else None
                    func_name = func.__name__
                    lock_name = getattr(self_obj, "lock_name", "unknown") if self_obj else "unknown"

                    if retry < max_retries - 1:
                        current_delay = retry_delay * (2**retry)
                        logging.warning(f"{func_name} {lock_name} failed: {str(e)}, retrying ({retry + 1}/{max_retries})")
                        time.sleep(current_delay)
                    else:
                        logging.error(f"{func_name} {lock_name} failed after all attempts: {str(e)}")

            if last_exception:
                raise last_exception
            return False

        return wrapper

    return decorator


class PostgresDatabaseLock:
    def __init__(self, lock_name, timeout=10, db=None):
        self.lock_name = lock_name
        self.lock_id = int(hashlib.md5(lock_name.encode()).hexdigest(), 16) % (2**31 - 1)
        self.timeout = int(timeout)
        self.db = db if db else DB

    @with_retry(max_retries=3, retry_delay=1.0)
    def lock(self):
        cursor = self.db.execute_sql("SELECT pg_try_advisory_lock(%s)", (self.lock_id,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"acquire postgres lock {self.lock_name} timeout")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"failed to acquire lock {self.lock_name}")

    @with_retry(max_retries=3, retry_delay=1.0)
    def unlock(self):
        cursor = self.db.execute_sql("SELECT pg_advisory_unlock(%s)", (self.lock_id,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"postgres lock {self.lock_name} was not established by this thread")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"postgres lock {self.lock_name} does not exist")

    def __enter__(self):
        if isinstance(self.db, PooledPostgresqlDatabase):
            self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.db, PooledPostgresqlDatabase):
            self.unlock()

    def __call__(self, func):
        @wraps(func)
        def magic(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return magic


class MysqlDatabaseLock:
    def __init__(self, lock_name, timeout=10, db=None):
        self.lock_name = lock_name
        self.timeout = int(timeout)
        self.db = db if db else DB

    @with_retry(max_retries=3, retry_delay=1.0)
    def lock(self):
        # SQL parameters only support %s format placeholders
        cursor = self.db.execute_sql("SELECT GET_LOCK(%s, %s)", (self.lock_name, self.timeout))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"acquire mysql lock {self.lock_name} timeout")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"failed to acquire lock {self.lock_name}")

    @with_retry(max_retries=3, retry_delay=1.0)
    def unlock(self):
        cursor = self.db.execute_sql("SELECT RELEASE_LOCK(%s)", (self.lock_name,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"mysql lock {self.lock_name} was not established by this thread")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"mysql lock {self.lock_name} does not exist")

    def __enter__(self):
        if isinstance(self.db, PooledMySQLDatabase):
            self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.db, PooledMySQLDatabase):
            self.unlock()

    def __call__(self, func):
        @wraps(func)
        def magic(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return magic


class DatabaseLock(Enum):
    MYSQL = MysqlDatabaseLock
    POSTGRES = PostgresDatabaseLock


DB = BaseDataBase().database_connection
DB.lock = DatabaseLock[settings.DATABASE_TYPE.upper()].value


def close_connection():
    try:
        if DB:
            DB.close_stale(age=30)
    except Exception as e:
        logging.exception(e)


class DataBaseModel(BaseModel):
    class Meta:
        database = DB


@DB.connection_context()
def init_database_tables(alter_fields=[]):
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    table_objs = []
    create_failed_list = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):
            table_objs.append(obj)

            if not obj.table_exists():
                logging.debug(f"start create table {obj.__name__}")
                try:
                    obj.create_table()
                    logging.debug(f"create table success: {obj.__name__}")
                except Exception as e:
                    logging.exception(e)
                    create_failed_list.append(obj.__name__)
            else:
                logging.debug(f"table {obj.__name__} already exists, skip creation.")

    if create_failed_list:
        logging.error(f"create tables failed: {create_failed_list}")
        raise Exception(f"create tables failed: {create_failed_list}")
    migrate_db()


def fill_db_model_object(model_object, human_model_dict):
    for k, v in human_model_dict.items():
        attr_name = "%s" % k
        if hasattr(model_object.__class__, attr_name):
            setattr(model_object, attr_name, v)
    return model_object


# 用户模型类，继承自DataBaseModel和UserMixin
class User(DataBaseModel, UserMixin):
    # 用户唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 用户访问令牌
    access_token = CharField(max_length=255, null=True, index=True)
    # 用户昵称
    nickname = CharField(max_length=100, null=False, help_text="nicky name", index=True)
    # 用户密码
    password = CharField(max_length=255, null=True, help_text="password", index=True)
    # 用户邮箱地址
    email = CharField(max_length=255, null=False, help_text="email", index=True)
    # 用户头像，Base64编码字符串
    avatar = TextField(null=True, help_text="avatar base64 string")
    # 用户语言设置，默认为中文或英文
    language = CharField(max_length=32, null=True, help_text="English|Chinese", default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", index=True)
    # 用户颜色主题设置
    color_schema = CharField(max_length=32, null=True, help_text="Bright|Dark", default="Bright", index=True)
    # 用户时区设置
    timezone = CharField(max_length=64, null=True, help_text="Timezone", default="UTC+8\tAsia/Shanghai", index=True)
    # 用户最后登录时间
    last_login_time = DateTimeField(null=True, index=True)
    # 用户是否已认证
    is_authenticated = CharField(max_length=1, null=False, default="1", index=True)
    # 用户是否激活
    is_active = CharField(max_length=1, null=False, default="1", index=True)
    # 用户是否匿名
    is_anonymous = CharField(max_length=1, null=False, default="0", index=True)
    # 用户登录渠道
    login_channel = CharField(null=True, help_text="from which user login", index=True)
    # 用户状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)
    # 用户是否为超级管理员
    is_superuser = BooleanField(null=True, help_text="is root", default=False, index=True)

    # 返回用户邮箱作为字符串表示
    def __str__(self):
        return self.email

    # 获取用户JWT令牌
    def get_id(self):
        jwt = Serializer(secret_key=settings.SECRET_KEY)
        return jwt.dumps(str(self.access_token))

    # 数据库表元数据
    class Meta:
        db_table = "user"


# 租户模型类
class Tenant(DataBaseModel):
    # 租户唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 租户名称
    name = CharField(max_length=100, null=True, help_text="Tenant name", index=True)
    # 租户公钥
    public_key = CharField(max_length=255, null=True, index=True)
    # 默认聊天模型ID
    llm_id = CharField(max_length=128, null=False, help_text="default llm ID", index=True)
    # 默认嵌入模型ID
    embd_id = CharField(max_length=128, null=False, help_text="default embedding model ID", index=True)
    # 默认语音识别模型ID
    asr_id = CharField(max_length=128, null=False, help_text="default ASR model ID", index=True)
    # 默认图像转文本模型ID
    img2txt_id = CharField(max_length=128, null=False, help_text="default image to text model ID", index=True)
    # 默认重排序模型ID
    rerank_id = CharField(max_length=128, null=False, help_text="default rerank model ID", index=True)
    # 默认语音合成模型ID
    tts_id = CharField(max_length=256, null=True, help_text="default tts model ID", index=True)
    # 文档处理器ID列表
    parser_ids = CharField(max_length=256, null=False, help_text="document processors", index=True)
    # 租户信用额度
    credit = IntegerField(default=512, index=True)
    # 租户状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 数据库表元数据
    class Meta:
        db_table = "tenant"


# 用户租户关联模型类
class UserTenant(DataBaseModel):
    # 关联记录唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 用户ID
    user_id = CharField(max_length=32, null=False, index=True)
    # 租户ID
    tenant_id = CharField(max_length=32, null=False, index=True)
    # 用户在租户中的角色
    role = CharField(max_length=32, null=False, help_text="UserTenantRole", index=True)
    # 邀请人ID
    invited_by = CharField(max_length=32, null=False, index=True)
    # 关联状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 数据库表元数据
    class Meta:
        db_table = "user_tenant"


class InvitationCode(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    code = CharField(max_length=32, null=False, index=True)
    visit_time = DateTimeField(null=True, index=True)
    user_id = CharField(max_length=32, null=True, index=True)
    tenant_id = CharField(max_length=32, null=True, index=True)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "invitation_code"


# LLM工厂模型类
class LLMFactories(DataBaseModel):
    # LLM工厂名称，主键
    name = CharField(max_length=128, null=False, help_text="LLM factory name", primary_key=True)
    # LLM工厂logo，Base64编码
    logo = TextField(null=True, help_text="llm logo base64")
    # LLM工厂标签，如LLM、Text Embedding、Image2Text、ASR
    tags = CharField(max_length=255, null=False, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    # 工厂状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 返回工厂名称作为字符串表示
    def __str__(self):
        return self.name

    # 数据库表元数据
    class Meta:
        db_table = "llm_factories"


# LLM模型类
class LLM(DataBaseModel):
    # LLM模型名称
    llm_name = CharField(max_length=128, null=False, help_text="LLM name", index=True)
    # 模型类型，如LLM、Text Embedding、Image2Text、ASR
    model_type = CharField(max_length=128, null=False, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    # 所属工厂ID
    fid = CharField(max_length=128, null=False, help_text="LLM factory id", index=True)
    # 最大token数量
    max_tokens = IntegerField(default=0)

    # 模型标签，如LLM、Text Embedding、Image2Text、Chat、32k等
    tags = CharField(max_length=255, null=False, help_text="LLM, Text Embedding, Image2Text, Chat, 32k...", index=True)
    # 是否支持工具调用
    is_tools = BooleanField(null=False, help_text="support tools", default=False)
    # 模型状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 返回模型名称作为字符串表示
    def __str__(self):
        return self.llm_name

    # 数据库表元数据，使用复合主键
    class Meta:
        primary_key = CompositeKey("fid", "llm_name")
        db_table = "llm"


# 租户LLM配置模型类
class TenantLLM(DataBaseModel):
    # 租户ID
    tenant_id = CharField(max_length=32, null=False, index=True)
    # LLM工厂名称
    llm_factory = CharField(max_length=128, null=False, help_text="LLM factory name", index=True)
    # 模型类型
    model_type = CharField(max_length=128, null=True, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    # LLM模型名称
    llm_name = CharField(max_length=128, null=True, help_text="LLM name", default="", index=True)
    # API密钥
    api_key = CharField(max_length=2048, null=True, help_text="API KEY", index=True)
    # API基础URL
    api_base = CharField(max_length=255, null=True, help_text="API Base")
    # 最大token数量
    max_tokens = IntegerField(default=8192, index=True)
    # 已使用token数量
    used_tokens = IntegerField(default=0, index=True)

    # 返回模型名称作为字符串表示
    def __str__(self):
        return self.llm_name

    # 数据库表元数据，使用复合主键
    class Meta:
        db_table = "tenant_llm"
        primary_key = CompositeKey("tenant_id", "llm_factory", "llm_name")


# 租户Langfuse配置模型类
class TenantLangfuse(DataBaseModel):
    # 租户ID，主键
    tenant_id = CharField(max_length=32, null=False, primary_key=True)
    # 密钥
    secret_key = CharField(max_length=2048, null=False, help_text="SECRET KEY", index=True)
    # 公钥
    public_key = CharField(max_length=2048, null=False, help_text="PUBLIC KEY", index=True)
    # 主机地址
    host = CharField(max_length=128, null=False, help_text="HOST", index=True)

    # 返回Langfuse主机信息作为字符串表示
    def __str__(self):
        return "Langfuse host" + self.host

    # 数据库表元数据
    class Meta:
        db_table = "tenant_langfuse"


# 知识库模型类
class Knowledgebase(DataBaseModel):
    # 知识库唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 知识库头像，Base64编码
    avatar = TextField(null=True, help_text="avatar base64 string")
    # 所属租户ID
    tenant_id = CharField(max_length=32, null=False, index=True)
    # 知识库名称
    name = CharField(max_length=128, null=False, help_text="KB name", index=True)
    # 知识库语言，默认为中文或英文
    language = CharField(max_length=32, null=True, default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", help_text="English|Chinese", index=True)
    # 知识库描述
    description = TextField(null=True, help_text="KB description")
    # 默认嵌入模型ID
    embd_id = CharField(max_length=128, null=False, help_text="default embedding model ID", index=True)
    # 权限设置，me表示个人，team表示团队
    permission = CharField(max_length=16, null=False, help_text="me|team", default="me", index=True)
    # 创建者ID
    created_by = CharField(max_length=32, null=False, index=True)
    # 文档数量
    doc_num = IntegerField(default=0, index=True)
    # token数量
    token_num = IntegerField(default=0, index=True)
    # 分块数量
    chunk_num = IntegerField(default=0, index=True)
    # 相似度阈值
    similarity_threshold = FloatField(default=0.2, index=True)
    # 向量相似度权重
    vector_similarity_weight = FloatField(default=0.3, index=True)

    # 默认解析器ID
    parser_id = CharField(max_length=32, null=False, help_text="default parser ID", default=ParserType.NAIVE.value, index=True)
    # 解析器配置，JSON格式
    parser_config = JSONField(null=False, default={"pages": [[1, 1000000]]})
    # 页面排名
    pagerank = IntegerField(default=0, index=False)
    # 知识库状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 返回知识库名称作为字符串表示
    def __str__(self):
        return self.name

    # 数据库表元数据
    class Meta:
        db_table = "knowledgebase"


# 文档模型类
class Document(DataBaseModel):
    # 文档唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 文档缩略图，Base64编码
    thumbnail = TextField(null=True, help_text="thumbnail base64 string")
    # 所属知识库ID
    kb_id = CharField(max_length=256, null=False, index=True)
    # 解析器ID
    parser_id = CharField(max_length=32, null=False, help_text="default parser ID", index=True)
    # 解析器配置，JSON格式
    parser_config = JSONField(null=False, default={"pages": [[1, 1000000]]})
    # 文档来源类型，默认为本地
    source_type = CharField(max_length=128, null=False, default="local", help_text="where dose this document come from", index=True)
    # 文档类型，文件扩展名
    type = CharField(max_length=32, null=False, help_text="file extension", index=True)
    # 创建者ID
    created_by = CharField(max_length=32, null=False, help_text="who created it", index=True)
    # 文档名称
    name = CharField(max_length=255, null=True, help_text="file name", index=True)
    # 文档存储位置
    location = CharField(max_length=255, null=True, help_text="where dose it store", index=True)
    # 文档大小
    size = IntegerField(default=0, index=True)
    # token数量
    token_num = IntegerField(default=0, index=True)
    # 分块数量
    chunk_num = IntegerField(default=0, index=True)
    # 处理进度
    progress = FloatField(default=0, index=True)
    # 处理进度消息
    progress_msg = TextField(null=True, help_text="process message", default="")
    # 处理开始时间
    process_begin_at = DateTimeField(null=True, index=True)
    # 处理持续时间
    process_duration = FloatField(default=0)
    # 元数据字段，JSON格式
    meta_fields = JSONField(null=True, default={})
    # 真实文件扩展名后缀
    suffix = CharField(max_length=32, null=False, help_text="The real file extension suffix", index=True)

    # 运行状态，1表示开始处理，2表示取消，0表示未开始
    run = CharField(max_length=1, null=True, help_text="start to run processing or cancel.(1: run it; 2: cancel)", default="0", index=True)
    # 文档状态，1表示有效，0表示无效
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    # 数据库表元数据
    class Meta:
        db_table = "document"


# 文件模型类
class File(DataBaseModel):
    # 文件唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 父文件夹ID
    parent_id = CharField(max_length=32, null=False, help_text="parent folder id", index=True)
    # 所属租户ID
    tenant_id = CharField(max_length=32, null=False, help_text="tenant id", index=True)
    # 创建者ID
    created_by = CharField(max_length=32, null=False, help_text="who created it", index=True)
    # 文件或文件夹名称
    name = CharField(max_length=255, null=False, help_text="file name or folder name", index=True)
    # 文件存储位置
    location = CharField(max_length=255, null=True, help_text="where dose it store", index=True)
    # 文件大小
    size = IntegerField(default=0, index=True)
    # 文件类型，扩展名
    type = CharField(max_length=32, null=False, help_text="file extension", index=True)
    # 文件来源类型
    source_type = CharField(max_length=128, null=False, default="", help_text="where dose this document come from", index=True)

    # 数据库表元数据
    class Meta:
        db_table = "file"


# 文件到文档关联模型类
class File2Document(DataBaseModel):
    # 关联记录唯一标识符，主键
    id = CharField(max_length=32, primary_key=True)
    # 文件ID
    file_id = CharField(max_length=32, null=True, help_text="file id", index=True)
    # 文档ID
    document_id = CharField(max_length=32, null=True, help_text="document id", index=True)

    # 数据库表元数据
    class Meta:
        db_table = "file2document"


class Task(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    doc_id = CharField(max_length=32, null=False, index=True)
    from_page = IntegerField(default=0)
    to_page = IntegerField(default=100000000)
    task_type = CharField(max_length=32, null=False, default="")
    priority = IntegerField(default=0)

    begin_at = DateTimeField(null=True, index=True)
    process_duration = FloatField(default=0)

    progress = FloatField(default=0, index=True)
    progress_msg = TextField(null=True, help_text="process message", default="")
    retry_count = IntegerField(default=0)
    digest = TextField(null=True, help_text="task digest", default="")
    chunk_ids = LongTextField(null=True, help_text="chunk ids", default="")


class Dialog(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    tenant_id = CharField(max_length=32, null=False, index=True)
    name = CharField(max_length=255, null=True, help_text="dialog application name", index=True)
    description = TextField(null=True, help_text="Dialog description")
    icon = TextField(null=True, help_text="icon base64 string")
    language = CharField(max_length=32, null=True, default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", help_text="English|Chinese", index=True)
    llm_id = CharField(max_length=128, null=False, help_text="default llm ID")

    llm_setting = JSONField(null=False, default={"temperature": 0.1, "top_p": 0.3, "frequency_penalty": 0.7, "presence_penalty": 0.4, "max_tokens": 512})
    prompt_type = CharField(max_length=16, null=False, default="simple", help_text="simple|advanced", index=True)
    prompt_config = JSONField(
        null=False,
        default={"system": "", "prologue": "Hi! I'm your assistant, what can I do for you?", "parameters": [], "empty_response": "Sorry! No relevant content was found in the knowledge base!"},
    )

    similarity_threshold = FloatField(default=0.2)
    vector_similarity_weight = FloatField(default=0.3)

    top_n = IntegerField(default=6)

    top_k = IntegerField(default=1024)

    do_refer = CharField(max_length=1, null=False, default="1", help_text="it needs to insert reference index into answer or not")

    rerank_id = CharField(max_length=128, null=False, help_text="default rerank model ID")

    kb_ids = JSONField(null=False, default=[])
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "dialog"


class Conversation(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    dialog_id = CharField(max_length=32, null=False, index=True)
    name = CharField(max_length=255, null=True, help_text="converastion name", index=True)
    message = JSONField(null=True)
    reference = JSONField(null=True, default=[])
    user_id = CharField(max_length=255, null=True, help_text="user_id", index=True)

    class Meta:
        db_table = "conversation"


class APIToken(DataBaseModel):
    tenant_id = CharField(max_length=32, null=False, index=True)
    token = CharField(max_length=255, null=False, index=True)
    dialog_id = CharField(max_length=32, null=True, index=True)
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)
    beta = CharField(max_length=255, null=True, index=True)

    class Meta:
        db_table = "api_token"
        primary_key = CompositeKey("tenant_id", "token")


class API4Conversation(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    dialog_id = CharField(max_length=32, null=False, index=True)
    user_id = CharField(max_length=255, null=False, help_text="user_id", index=True)
    message = JSONField(null=True)
    reference = JSONField(null=True, default=[])
    tokens = IntegerField(default=0)
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)
    dsl = JSONField(null=True, default={})
    duration = FloatField(default=0, index=True)
    round = IntegerField(default=0, index=True)
    thumb_up = IntegerField(default=0, index=True)

    class Meta:
        db_table = "api_4_conversation"


class UserCanvas(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    avatar = TextField(null=True, help_text="avatar base64 string")
    user_id = CharField(max_length=255, null=False, help_text="user_id", index=True)
    title = CharField(max_length=255, null=True, help_text="Canvas title")

    permission = CharField(max_length=16, null=False, help_text="me|team", default="me", index=True)
    description = TextField(null=True, help_text="Canvas description")
    canvas_type = CharField(max_length=32, null=True, help_text="Canvas type", index=True)
    dsl = JSONField(null=True, default={})

    class Meta:
        db_table = "user_canvas"


class CanvasTemplate(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    avatar = TextField(null=True, help_text="avatar base64 string")
    title = CharField(max_length=255, null=True, help_text="Canvas title")

    description = TextField(null=True, help_text="Canvas description")
    canvas_type = CharField(max_length=32, null=True, help_text="Canvas type", index=True)
    dsl = JSONField(null=True, default={})

    class Meta:
        db_table = "canvas_template"


class UserCanvasVersion(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    user_canvas_id = CharField(max_length=255, null=False, help_text="user_canvas_id", index=True)

    title = CharField(max_length=255, null=True, help_text="Canvas title")
    description = TextField(null=True, help_text="Canvas description")
    dsl = JSONField(null=True, default={})

    class Meta:
        db_table = "user_canvas_version"


class MCPServer(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    name = CharField(max_length=255, null=False, help_text="MCP Server name")
    tenant_id = CharField(max_length=32, null=False, index=True)
    url = CharField(max_length=2048, null=False, help_text="MCP Server URL")
    server_type = CharField(max_length=32, null=False, help_text="MCP Server type")
    description = TextField(null=True, help_text="MCP Server description")
    variables = JSONField(null=True, default=dict, help_text="MCP Server variables")
    headers = JSONField(null=True, default=dict, help_text="MCP Server additional request headers")

    class Meta:
        db_table = "mcp_server"


class Search(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    avatar = TextField(null=True, help_text="avatar base64 string")
    tenant_id = CharField(max_length=32, null=False, index=True)
    name = CharField(max_length=128, null=False, help_text="Search name", index=True)
    description = TextField(null=True, help_text="KB description")
    created_by = CharField(max_length=32, null=False, index=True)
    search_config = JSONField(
        null=False,
        default={
            "kb_ids": [],
            "doc_ids": [],
            "similarity_threshold": 0.0,
            "vector_similarity_weight": 0.3,
            "use_kg": False,
            # rerank settings
            "rerank_id": "",
            "top_k": 1024,
            # chat settings
            "summary": False,
            "chat_id": "",
            "llm_setting": {
                "temperature": 0.1,
                "top_p": 0.3,
                "frequency_penalty": 0.7,
                "presence_penalty": 0.4,
            },
            "chat_settingcross_languages": [],
            "highlight": False,
            "keyword": False,
            "web_search": False,
            "related_search": False,
            "query_mindmap": False,
        },
    )
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "search"


def migrate_db():
    migrator = DatabaseMigrator[settings.DATABASE_TYPE.upper()].value(DB)
    try:
        migrate(migrator.add_column("file", "source_type", CharField(max_length=128, null=False, default="", help_text="where dose this document come from", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("tenant", "rerank_id", CharField(max_length=128, null=False, default="BAAI/bge-reranker-v2-m3", help_text="default rerank model ID")))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("dialog", "rerank_id", CharField(max_length=128, null=False, default="", help_text="default rerank model ID")))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("dialog", "top_k", IntegerField(default=1024)))
    except Exception:
        pass
    try:
        migrate(migrator.alter_column_type("tenant_llm", "api_key", CharField(max_length=2048, null=True, help_text="API KEY", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("api_token", "source", CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("tenant", "tts_id", CharField(max_length=256, null=True, help_text="default tts model ID", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("api_4_conversation", "source", CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("task", "retry_count", IntegerField(default=0)))
    except Exception:
        pass
    try:
        migrate(migrator.alter_column_type("api_token", "dialog_id", CharField(max_length=32, null=True, index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("tenant_llm", "max_tokens", IntegerField(default=8192, index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("api_4_conversation", "dsl", JSONField(null=True, default={})))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("knowledgebase", "pagerank", IntegerField(default=0, index=False)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("api_token", "beta", CharField(max_length=255, null=True, index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("task", "digest", TextField(null=True, help_text="task digest", default="")))
    except Exception:
        pass

    try:
        migrate(migrator.add_column("task", "chunk_ids", LongTextField(null=True, help_text="chunk ids", default="")))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("conversation", "user_id", CharField(max_length=255, null=True, help_text="user_id", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("document", "meta_fields", JSONField(null=True, default={})))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("task", "task_type", CharField(max_length=32, null=False, default="")))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("task", "priority", IntegerField(default=0)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("user_canvas", "permission", CharField(max_length=16, null=False, help_text="me|team", default="me", index=True)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("llm", "is_tools", BooleanField(null=False, help_text="support tools", default=False)))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("mcp_server", "variables", JSONField(null=True, help_text="MCP Server variables", default=dict)))
    except Exception:
        pass
    try:
        migrate(migrator.rename_column("task", "process_duation", "process_duration"))
    except Exception:
        pass
    try:
        migrate(migrator.rename_column("document", "process_duation", "process_duration"))
    except Exception:
        pass
    try:
        migrate(migrator.add_column("document", "suffix", CharField(max_length=32, null=False, default="", help_text="The real file extension suffix", index=True)))
    except Exception:
        pass
