# backend/memory.py

class ConversationMemory:
    """
    对话记忆管理器
    
    为什么做成一个类而不是用全局列表？
    因为未来每个用户应该有自己独立的记忆
    用类封装方便后续扩展成 user_id → memory 的字典
    """
    
    def __init__(self, strategy: str = "window", window_size: int = 6):
        """
        strategy: "full" | "window" | "summary"
        window_size: 滑动窗口保留最近几条消息（注意是消息数，不是轮数）
        """
        self.strategy = strategy
        self.window_size = window_size
        self.messages = []  # 完整历史，永远保留
        self.summary = ""   # 摘要策略用
    
    def add(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})
    
    def get_context(self) -> list:
        """
        根据策略返回要发给 LLM 的消息列表
        这是记忆管理的核心方法
        """
        if self.strategy == "full":
            return self._full()
        elif self.strategy == "window":
            return self._window()
        elif self.strategy == "summary":
            return self._summary()
        return self.messages
    
    def _full(self) -> list:
        """
        策略一：全量历史
        优点：LLM 记得所有内容，回答最准确
        缺点：对话越长 token 越多越贵
              10轮对话可能已经用掉 3000 token，100轮就是灾难
        适用：短会话、测试阶段
        """
        return self.messages
    
    def _window(self) -> list:
        """
        策略二：滑动窗口
        只取最近 window_size 条消息
        
        为什么是 window_size=6（3轮）而不是更多？
        经验值：大多数对话里，用户当前问题和最近2-3轮最相关
        更早的内容影响力递减，带着反而引入噪音
        
        缺点：如果用户在第1轮说了重要信息（比如目标城市），
              第10轮之后 LLM 就忘了
        """
        return self.messages[-self.window_size:]
    
    def _summary(self) -> list:
        """
        策略三：摘要压缩
        把旧对话压缩成一段摘要，再加上最近几轮
        
        原理：
        [旧对话20轮] → LLM压缩 → [摘要1段]
        [摘要] + [最近3轮] → 发给LLM
        
        优点：既不丢失重要信息，又控制了 token 数
        缺点：需要额外调用 LLM 来生成摘要，有延迟和费用
        适用：长会话、生产环境
        
        现在先返回 window 版本作为占位，
        真实摘要实现需要异步调用 LLM，是进阶功能
        """
        if self.summary:
            summary_msg = {
                "role": "system",
                "content": f"以下是之前对话的摘要：{self.summary}"
            }
            return [summary_msg] + self.messages[-4:]
        return self._window()
    
    def clear(self):
        """清空记忆，开始新对话"""
        self.messages = []
        self.summary = ""
    
    @property
    def turn_count(self) -> int:
        """对话轮数，每轮 = 1个human + 1个assistant"""
        return len(self.messages) // 2