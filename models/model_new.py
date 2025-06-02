# models/model.py

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTMEncoder: 单向 LSTM，将输入序列 (B, seq_len, input_dim) 映射到隐藏态
    hidden_dim 默认 64，对齐图示。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,       # e.g. 6
            hidden_size=hidden_dim,     # e.g. 64
            num_layers=num_layers,      # e.g. 2
            batch_first=True
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [B, seq_len, input_dim]
        Returns:
            outputs: Tensor of shape [B, seq_len, hidden_dim]
            (h_n, c_n): each of shape [num_layers, B, hidden_dim]
        """
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """
    LSTMDecoder: 
      - 首先把输入 (B, 1, 2) 送入 LSTM → 得到 (B, 1, hidden_dim)
      - 再通过 Linear(hidden_dim → 2) 得到当前时刻的预测 (B, 1, 2)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,       # 2
            hidden_size=hidden_dim,     # 64
            num_layers=num_layers,      # 2
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: tuple):
        """
        Args:
            x: Tensor of shape [B, 1, input_dim]  （每步只输入一个时间步）
            hidden: tuple (h_n, c_n)，
                h_n: Tensor [num_layers, B, hidden_dim]
                c_n: Tensor [num_layers, B, hidden_dim]
        Returns:
            out2: Tensor of shape [B, 1, output_dim]
            (h_n, c_n): 更新后的 hidden 状态
        """
        out, (h_n, c_n) = self.lstm(x, hidden)
        # out: [B, 1, hidden_dim]
        out2 = self.linear(out)  # [B, 1, output_dim]
        return out2, (h_n, c_n)


class Seq2SeqLSTMModel(nn.Module):
    """
    序列到序列 LSTM 模型，用于轨迹预测
    --------
    1) Encoder LSTM(6→64) → 得到最后一层的 hidden, cell
    2) Learnable Start Token (1×1×2) → expand 到 (B, 1, 2)
    3) 循环 output_seq_len 次：
       - Decoder LSTM(2→64) → Linear(64→2) → 得到当前时刻预测 (B,1,2)
       - 根据 teacher_forcing 决定下一步输入：上一预测 or 真实值
    4) 最终输出 shape=(B, output_seq_len, 2)
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_seq_len: int = 60,
        output_dim: int = 2,
        num_layers: int = 2
    ):
        super(Seq2SeqLSTMModel, self).__init__()

        # 1) Encoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # 2) Decoder
        self.decoder = LSTMDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )

        # 3) Learnable Start Token: 初始化为 (1,1,2)，后续 expand 到 (B,1,2)
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.1)

        # 保存超参数
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, data: dict, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        执行一次正向传播，逐步生成 output_seq_len 帧预测
        
        Args:
            data: 字典，包含
                - history: Tensor [B, num_agents, seq_len, feature_dim]  （6）
                - future: Tensor [B, output_seq_len, output_dim]（仅训练时用于 teacher forcing）
                - scale_position: Tensor [B] 或 [B,] 
            teacher_forcing_ratio: float in [0,1]，
                - 若 > 0 且在训练模式下，会以该概率选择用 ground-truth 作为下一步输入。

        Returns:
            predictions: Tensor of shape [B, output_seq_len, output_dim]
        """
        history = data["history"]               # [B, num_agents, seq_len, 6]
        batch_size = history.size(0)

        # ----- 只取 Ego agent 的历史: shape → [B, seq_len, 6]
        ego_history = history[:, 0, :, :]       # [B, 50, 6]

        # ----- Encoder: 得到 hidden, cell
        _, (h_n, c_n) = self.encoder(ego_history)
        # h_n, c_n: [num_layers, B, hidden_dim]

        # 判断是否使用 teacher forcing
        use_teacher_forcing = (
            self.training
            and ("future" in data)
            and (torch.rand(1).item() < teacher_forcing_ratio)
        )

        # ----- Decoder 第一个输入: learned start token
        # start_token: [1,1,2] → expand → [B,1,2]
        decoder_input = self.start_token.expand(batch_size, -1, -1).to(history.device)
        hidden = (h_n, c_n)

        # 预分配 predictions 张量
        predictions = torch.zeros(
            batch_size,
            self.output_seq_len,
            self.output_dim,
            device=history.device
        )

        # ----- 逐步 decode
        for t in range(self.output_seq_len):
            # （1）Decoder LSTM + Linear
            output, hidden = self.decoder(decoder_input, hidden)
            # output: [B,1,2]

            # （2）保存当前时刻预测
            predictions[:, t : t + 1, :] = output

            # （3）根据 Teacher Forcing 决定下一步输入
            if use_teacher_forcing and t < self.output_seq_len - 1:
                # 训练时，若使用 Teacher Forcing，用真实 future
                decoder_input = data["future"][:, t : t + 1, :].to(history.device)
            else:
                # 否则使用模型自己的输出
                decoder_input = output

        return predictions
