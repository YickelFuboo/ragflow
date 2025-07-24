# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""
import re
from typing import Any
from dataclasses import dataclass
from graphrag.general.extractor import Extractor, ENTITY_EXTRACTION_MAX_GLEANINGS
from graphrag.light.graph_prompt import PROMPTS
from graphrag.utils import pack_user_ass_to_openai_messages, split_string_by_multi_markers, chat_limiter
from rag.llm.chat_model import Base as CompletionLLM
import networkx as nx
from rag.utils import num_tokens_from_string
import trio


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor(Extractor):

    _max_gleanings: int

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        example_number: int = 2,
        max_gleanings: int | None = None,
    ):
        super().__init__(llm_invoker, language, entity_types)
        """Init method definition."""
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        self._example_number = example_number
        examples = "\n".join(
                PROMPTS["entity_extraction_examples"][: int(self._example_number)]
            )

        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            language=self._language,
        )
        # add example's format
        examples = examples.format(**example_context_base)

        self._entity_extract_prompt = PROMPTS["entity_extraction"]
        self._context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            examples=examples,
            language=self._language,
        )

        self._continue_prompt = PROMPTS["entiti_continue_extraction"]
        self._if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

        self._left_token_count = llm_invoker.max_length - num_tokens_from_string(
            self._entity_extract_prompt.format(
                **self._context_base, input_text="{input_text}"
            ).format(**self._context_base, input_text="")
        )
        self._left_token_count = max(llm_invoker.max_length * 0.6, self._left_token_count)

    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results):
        """
        处理单个文档切片，提取实体和关系
        
        Args:
            chunk_key_dp: 包含文档ID和切片内容的元组 (doc_id, content)
            chunk_seq: 当前切片在文档中的序号
            num_chunks: 文档总切片数
            out_results: 存储所有处理结果的列表
            
        Returns:
            无返回值，结果存储在out_results中
        """
        # 初始化token计数器，用于跟踪LLM调用消耗的token数量
        token_count = 0
        chunk_key = chunk_key_dp[0]  # 提取文档ID
        content = chunk_key_dp[1]    # 提取切片内容
        
        # 构建实体提取的提示词，将切片内容嵌入到预定义的模板中
        hint_prompt = self._entity_extract_prompt.format(
            **self._context_base, input_text="{input_text}"
        ).format(**self._context_base, input_text=content)

        # 设置LLM生成配置，使用较高的温度值以增加输出的多样性
        gen_conf = {"temperature": 0.8}
        
        # 第一次调用LLM进行实体提取，使用聊天限制器控制并发
        async with chat_limiter:
            final_result = await trio.to_thread.run_sync(lambda: self._chat(hint_prompt, [{"role": "user", "content": "Output:"}], gen_conf))
        
        # 累计消耗的token数量
        token_count += num_tokens_from_string(hint_prompt + final_result)
        
        # 构建对话历史，为后续的继续提取做准备
        history = pack_user_ass_to_openai_messages("Output:", final_result, self._continue_prompt)
        
        # 循环进行多次提取（最多max_gleanings次），以获取更完整的实体和关系
        for now_glean_index in range(self._max_gleanings):
            # 调用LLM继续提取更多实体和关系
            async with chat_limiter:
                glean_result = await trio.to_thread.run_sync(lambda: self._chat(hint_prompt, history, gen_conf))
            
            # 更新对话历史，添加助手的回复和用户的继续提示
            history.extend([{"role": "assistant", "content": glean_result}, {"role": "user", "content": self._continue_prompt}])
            
            # 累计token消耗
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + hint_prompt + self._continue_prompt)
            
            # 将新提取的结果追加到最终结果中
            final_result += glean_result
            
            # 如果已达到最大提取次数，则退出循环
            if now_glean_index == self._max_gleanings - 1:
                break

            # 询问LLM是否还需要继续提取
            async with chat_limiter:
                if_loop_result = await trio.to_thread.run_sync(lambda: self._chat(self._if_loop_prompt, history, gen_conf))
            
            # 累计token消耗
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + if_loop_result + self._if_loop_prompt)
            
            # 解析LLM的回复，判断是否需要继续提取
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # 解析LLM返回的结果，按记录分隔符分割
        records = split_string_by_multi_markers(
            final_result,
            [self._context_base["record_delimiter"], self._context_base["completion_delimiter"]],
        )
        
        # 提取括号内的内容，过滤掉格式不正确的记录
        rcds = []
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            rcds.append(record.group(1))
        records = rcds
        
        # 将解析后的记录转换为实体和关系
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, self._context_base["tuple_delimiter"])
        
        # 将处理结果添加到输出列表中
        out_results.append((maybe_nodes, maybe_edges, token_count))
        
        # 如果设置了回调函数，报告处理进度
        if self.callback:
            self.callback(0.5+0.1*len(out_results)/num_chunks, msg = f"Entities extraction of chunk {chunk_seq} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.")
