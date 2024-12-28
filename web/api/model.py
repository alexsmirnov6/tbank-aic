from transformers import Wav2Vec2Config, Wav2Vec2Model
import torch
import torch.nn as nn

class DisordersDetector(nn.Module):

    def __init__(self, cfg, stage):
        super().__init__()
        self.cfg = cfg
        if stage == "pretrain":
            self.backbone = Wav2Vec2Model.from_pretrained(cfg.model_name)
        else:
            # Не подгружаем модель с HF, ведь мы подгрузим претрейн модель
            model_cfg = Wav2Vec2Config.from_pretrained(cfg.model_name)
            self.backbone = Wav2Vec2Model(model_cfg)
            
        self.stage = stage

        dropout = cfg.dropout
        hidden_dim = 1024
        head_dim = cfg.head_dim

        if stage == "pretrain":
            self.letter_count_heads = nn.ModuleDict({
                f"{letter}_count_head": nn.Sequential(
                    nn.Linear(hidden_dim, head_dim),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, cfg.letters_num_classes[letter])
                )
                for letter in cfg.target_letters
            })

        else:
            self.disorders_head = nn.Sequential(nn.Linear(hidden_dim, head_dim),
                                            nn.Dropout(dropout),
                                            nn.Linear(head_dim, len(cfg.disorders_class_weights)))

    def forward(self, x):

        hidden_state = self.backbone(x).last_hidden_state
        pooled_output = torch.mean(hidden_state, dim=1)

        output = {}

        if self.stage == 'pretrain':
            for letter_head, head in self.letter_count_heads.items():
                output[letter_head[0]] = head(pooled_output)

        else:

            output['disorders'] = self.disorders_head(pooled_output)

        return output

    def freeze_feature_extractor(self):
        self.backbone.feature_extractor._freeze_parameters()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
            for param in self.backbone.parameters():
                param.requires_grad = True

            if self.cfg.model_type == "wav2vec":
                self.freeze_feature_extractor()
