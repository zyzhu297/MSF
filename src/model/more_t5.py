import sys
sys.path.append("/")

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast, AutoConfig, AutoModel
import numpy as np
import contextlib
# from .registry import registry
from transformers import PreTrainedModel
from transformers import T5ForConditionalGeneration

from .integrator import BertModel
from src.model.blip2_qformer import Blip2Qformer
import random
from .utils import *

# @registry.register_model("ke_t5")
class BaseMore(PreTrainedModel):
    def __init__(
        self,
        config,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def _normalize(self, feature, dim=-1):
        norm = feature.norm(p=2, dim=dim, keepdim=True)
        feature = feature.div(norm + 1e-8)
        return feature

    

    def init_Qformer(self, integrator_path, num_query_token, num_integrator_layers=2, integrator_attn_list=[["self", "cross"], ["self", "cross"]], num_former_layers=1, former_attn_list=[["cross"]]):
        integrator_embedding = Blip2Qformer(
            vision_path="src/pre_trained_lm/eva_vit_g.pth",
            former_path="src/pre_trained_lm/bert-base-uncased",
        )
        msg = integrator_embedding.load_from_pretrained("src/pre_trained_lm/blip2_pretrained.pth")
        integrator_embedding.eval()
        integrator_embedding = integrator_embedding.Qformer.bert.embeddings
        for name, param in integrator_embedding.named_parameters():
            param.requires_grad = False

        integrator_config = AutoConfig.from_pretrained(integrator_path)
        integrator_config.hidden_size = 768
        integrator_config.encoder_width = integrator_config.hidden_size
        # insert cross-attention layer every other block
        # integrator_config.query_length = num_query_token
        integrator_config.num_hidden_layers = num_integrator_layers
        integrator = BertModel(integrator_config, attn_list=[["self", "cross"], ["self"]])
        integrator.apply(self._init_weights)

        former_config = AutoConfig.from_pretrained(integrator_path)
        former_config.hidden_size = 768
        former_config.encoder_width = former_config.hidden_size
        # former_config.query_length = num_query_token
        former_config.num_hidden_layers = num_former_layers
        former = BertModel(former_config, attn_list=[["cross"]])
        former.apply(self._init_weights)

        query_tokens = InputPrompts(num_query_token, former_config.hidden_size)

        lm_proj = nn.Linear(
            former_config.hidden_size, self.backbone_model.config.hidden_size
        )
        
        lm_proj.apply(self._init_weights)
           
        return integrator_embedding, integrator, former, query_tokens, lm_proj

    def forward_image(
            self,
            image_inputs=None,
            image_inputs_attention_mask=None,
            device=None,
        ):
        image_embeds, image_atts = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        if image_inputs is not None and self.num_query_token>0:
            with self.maybe_autocast():
                # print(image_inputs)
                # input()
                bsz, num, l, hid = image_inputs.size()
                image_embeds = image_inputs.view(bsz, -1, hid)

                if self.image_proj is not None:
                    image_embeds = self.image_proj(image_embeds) # (bsz, N*l, h)
                image_atts = image_inputs_attention_mask.view(bsz, -1)

        return image_embeds, image_atts

    def forward_text(
            self,
            text_inputs=None,
            text_inputs_attention_mask=None,
            device=None,
        ):
        text_embeds, text_atts = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        if text_inputs is not None and self.num_query_token>0:
            bsz, n, l = text_inputs.size()           
            with self.maybe_autocast():
                bsz, l, hid = text_inputs.size()
                text_embeds = text_inputs.view(bsz, -1, hid)
                if self.text_proj is not None:
                    text_embeds = self.text_proj(text_embeds) # (bsz, N*l, h)
                text_atts = text_inputs_attention_mask.view(bsz, -1)
        
        return text_embeds, text_atts

    def forward_query(
            self,
            embeds=None,
            atts=None,
            former_text_input_ids=None,
            former_text_input_masks=None,
            device=None,
            integrator_embedding=None, 
            integrator=None, 
            former=None, 
            query_tokens=None,
            lm_proj=None,
        ):
        retrieved_inputs_t5, retrieved_atts_t5 = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        integrator_embedding = self.integrator_embedding if integrator_embedding is None else integrator_embedding
        integrator = self.integrator if integrator is None else integrator
        former = self.former if former is None else former
        query_tokens = self.query_tokens if query_tokens is None else query_tokens 
        lm_proj = self.lm_proj if lm_proj is None else lm_proj

        # print(former_text_input_ids.shape)
        if len(embeds) > 0:
            input_embeds = integrator_embedding(former_text_input_ids)
            # input query retrieved data
            integrator_output = integrator(
                query_embeds=input_embeds,
                attention_mask=former_text_input_masks,
                encoder_hidden_states=embeds,
                encoder_attention_mask=atts,
                return_dict=True,
            )

            # prompt query integrated data
            query_token_embeds = query_tokens(embeds.shape[0], device)
            encoder_attention_mask = torch.ones(integrator_output.last_hidden_state.size()[:-1], dtype=torch.long).to(device)
            former_output = former(
                query_embeds=query_token_embeds,
                encoder_hidden_states=integrator_output.last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )

            retrieved_inputs_t5 = lm_proj(former_output.last_hidden_state)
            retrieved_atts_t5 = torch.ones(retrieved_inputs_t5.size()[:-1], dtype=torch.long).to(device)

        return retrieved_inputs_t5, retrieved_atts_t5
        

    def get_input_embeddings(self):
        return self.backbone_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.backbone_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone_model.get_output_embeddings()

    def get_encoder(self):
        return self.backbone_model.get_encoder()

    def get_decoder(self):
        return self.backbone_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_backbone_model:
            self.backbone_model.encoder.embed_tokens = self.backbone_model.shared
            self.backbone_model.decoder.embed_tokens = self.backbone_model.shared


class MoreT5(BaseMore):
    def __init__(
        self,
        config,
        args,
        num_query_token=32,
        prompt_length=32,
        integrator_path="bert-base-uncased",
        backbone_model="google/flan-t5-xl",
        resume_checkpoint=None,   
        tokenizer=None,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__(config)
        print("MoreT5!")

        self.args = args
        self.num_query_token = num_query_token
        self.tokenizer = tokenizer

        ########## lm ##########
        self.backbone_model = T5ForConditionalGeneration.from_pretrained(
            backbone_model
        )
        self.generation_config = self.backbone_model.generation_config
        for name, param in self.backbone_model.named_parameters():
            param.requires_grad = False
            # param.data = param.data.bfloat16()
        self.backbone_model.eval()
        logging.info("freeze backbone language model")
        
        self.prompt_tokens = InputPrompts(prompt_length, self.backbone_model.config.hidden_size)

        ########## former ###########
        self.integrator_embedding, self.integrator, self.former, self.query_tokens, self.lm_proj, self.image_proj, self.text_proj = None, None, None, None, None, None, None
        if self.num_query_token > 0:
            self.integrator_embedding, self.integrator, self.former, self.query_tokens, self.lm_proj = self.init_Qformer(
                integrator_path, self.num_query_token, num_integrator_layers=2, num_former_layers=1
            )

            ########## image ##########
            if args.use_image and args.image_grounding_path is not None:
                self.image_proj = nn.Linear(768, self.integrator.config.hidden_size)
                self.image_proj.apply(self._init_weights)

            ########## text ###########
            if args.use_text:
                self.text_proj = nn.Linear(768, self.integrator.config.hidden_size)
                self.text_proj.apply(self._init_weights)
            
        if resume_checkpoint is not None:
            logging.info("Init pretrained model from pretrained_model_path: " + resume_checkpoint)
            model_dict = torch.load(resume_checkpoint+"/pytorch_model.bin")
            incompatible_keys = self.load_state_dict(model_dict, strict=False)
            print(incompatible_keys)
            

        train_param = 0
        all_param = 0
        module_param = {}
        print("====== tuning ======")
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad == True:
                train_param += param.numel()
                print(name)
                module_name = name.split(".")[0]
                if module_name not in module_param:
                    module_param[module_name] = 0
                module_param[module_name] += param.numel()
        print("====================")

        logging.info('total param is {}'.format(all_param)) # 9860105
        logging.info('train param is {}'.format(train_param)) # 9860105
        for k, v in module_param.items():
            logging.info('train {} param is {}'.format(k, v)) # 9860105

    def forward_query(
            self,
            embeds=None,
            atts=None,
            former_text_input_ids=None,
            former_text_input_masks=None,
            device=None,
            integrator_embedding=None, 
            integrator=None, 
            former=None, 
            query_tokens=None,
            lm_proj=None,
        ):
        retrieved_inputs_t5, retrieved_atts_t5 = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        integrator_embedding = self.integrator_embedding if integrator_embedding is None else integrator_embedding
        integrator = self.integrator if integrator is None else integrator
        former = self.former if former is None else former
        query_tokens = self.query_tokens if query_tokens is None else query_tokens 
        lm_proj = self.lm_proj if lm_proj is None else lm_proj

        # print(former_text_input_ids.shape)
        if len(embeds) > 0:
            input_embeds = integrator_embedding(former_text_input_ids)

            if self.args.whiteboard is not None:
                if self.args.whiteboard == "zero":
                    blank = torch.zeros([embeds.size(0), 1, embeds.size(-1)]).to(device)
                    blank_attn = torch.ones([embeds.size(0), 1]).long().to(device)
                elif self.args.whiteboard == "query":
                    blank = input_embeds.clone()
                    blank_attn = former_text_input_masks.clone()
                embeds = torch.cat([blank, embeds], 1)
                atts = torch.cat([blank_attn, atts], 1)


            # input query retrieved data
            integrator_output = integrator(
                query_embeds=input_embeds,
                attention_mask=former_text_input_masks,
                encoder_hidden_states=embeds,
                encoder_attention_mask=atts,
                return_dict=True,
            )

            # prompt query integrated data
            query_token_embeds = query_tokens(embeds.shape[0], device)
            encoder_attention_mask = torch.ones(integrator_output.last_hidden_state.size()[:-1], dtype=torch.long).to(device)
            former_output = former(
                query_embeds=query_token_embeds,
                encoder_hidden_states=integrator_output.last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )

            retrieved_inputs_t5 = lm_proj(former_output.last_hidden_state)
            retrieved_atts_t5 = torch.ones(retrieved_inputs_t5.size()[:-1], dtype=torch.long).to(device)

        return retrieved_inputs_t5, retrieved_atts_t5
        
    def forward(
        self, 
        input_ids=None,
        # raw_text_input=None,
        attention_mask=None,
        labels=None,
        label_attention_mask=None,
        image_inputs=None,
        image_inputs_attention_mask=None,
        text_inputs=None,
        text_inputs_attention_mask=None,
        former_text_input_ids=None,
        former_text_input_masks=None,
        cold_start_p=0,
        origin_image_feature=None,
        ):
        # import inspect
        # # import torch

        # frame = inspect.currentframe()
        # args, _, _, values = inspect.getargvalues(frame)
        
        # for arg in args:
        #     if arg == "self":
        #         continue
        #     value = values[arg]
        #     if isinstance(value, torch.Tensor):
        #         print(f"{arg}: {value.shape}")
        #     else:
        #         print(f"{arg}: {type(value)} | {value}")
        # print(input_ids)
        device = input_ids.device
        
        # print(device)
        # input()
        bsz = input_ids.shape[0]            
        image_features = origin_image_feature
        # self.projector = nn.Linear(768, 512)
        # if image_features is not None:
        #     image_embedding = image_features.mean(dim=1)  # 形状 (batch, 768)
        #     image_embedding_projected = self.projector(image_embedding)  # (batch, 512)

        ########## image ##########
        image_embeds, image_atts = self.forward_image(image_inputs, image_inputs_attention_mask, device)

        ########## text ##########
        text_embeds, text_atts = self.forward_text(text_inputs, text_inputs_attention_mask, device)

        # print('idjr', image_embeds.device, text_embeds.device, image_atts.device, text_atts.device)
        ########## query ##########
        retrieved_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        retrieved_atts = torch.cat([image_atts, text_atts], dim=1)
        # print('retsad', retrieved_embeds.device, retrieved_atts.device)
        # print('cold_start_p', cold_start_p)
        # print('random_p', self.args.random_p)
        # print(labels)
        if cold_start_p > 0:
            cold_start_probability_matrix = torch.full([bsz], cold_start_p)
            cold_start_indices_ = torch.bernoulli(cold_start_probability_matrix).bool() # bsz
            cold_start_indices = cold_start_indices_.unsqueeze(1).expand(-1, input_ids.size(1)).to(device) # bsz, l
            input_ids[cold_start_indices] = 0
            attention_mask[cold_start_indices] = 0

            if self.args.random_p > 0:
                rand_probability_matrix = torch.full([bsz], self.args.random_p)
                rand_indices = torch.bernoulli(rand_probability_matrix).bool() # bsz
                rand_indices = rand_indices * cold_start_indices_
                rand_indices = rand_indices.to(device)

                rand_retrieved_embeds = torch.cat([retrieved_embeds[:1], retrieved_embeds[1:]])
                rand_retrieved_atts = torch.cat([retrieved_atts[:1], retrieved_atts[1:]])

                retrieved_embeds[rand_indices] = rand_retrieved_embeds[rand_indices]
                retrieved_atts[rand_indices] = rand_retrieved_atts[rand_indices]
                #这里污染了label
                labels[rand_indices, 0] = 1
                labels[rand_indices, 1:] = 0
                # print(labels, rand_indices)

                label_attention_mask[rand_indices, 0] = 1
                label_attention_mask[rand_indices, 1:] = 0

        # print('input_ids')
        # print(input_ids[0])
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # 将 tokens 转换回原文
        # decoded_text = self.tokenizer.decode(input_ids[0])

        # 输出 ids 和 tokens 的对应关系
        # id_token_mapping = list(zip(input_ids[0], tokens))

        # print("原文:", decoded_text)
        # print("ID-Token 对应关系:", id_token_mapping)
        # print(input_ids)
        # print('image_features')
        # print(image_features)
        # print('-=-=' * 15)
        img_token_id = self.tokenizer.convert_tokens_to_ids("[IMG]")
        # print('img_token_id', img_token_id)
        # img_token_index = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
        img_positions = (input_ids == img_token_id).unsqueeze(-1).float()
        # print('img_pos', img_positions)
        # print('img_fe', image_features.shape, image_features) 
        if len(image_features.shape) == 5:
            pooled_image_features = [feat.mean(dim=2).squeeze(0) for feat in image_features]  # List[(768,)]

            # 转成一个 tensor，变成 (batch_size, 768)
            image_feature_broadcasted = torch.stack(pooled_image_features, dim=0)
            # print('imge_deda', image_feature_tensor.shape)
            # image_feature_broadcasted = image_feature_tensor.unsqueeze(1)
            # print('imge_berosdeda', image_feature_broadcasted.shape)
        else:
            pooled_image_features = [feat.mean(dim=1).squeeze(0) for feat in image_features]  # List[(768,)]

            # 转成一个 tensor，变成 (batch_size, 768)
            image_feature_tensor = torch.stack(pooled_image_features, dim=0)
            # print('imge_deda', image_feature_tensor.shape)
            image_feature_broadcasted = image_feature_tensor.unsqueeze(1)
            # print('imge_berosdeda', image_feature_broadcasted.shape)
        # print('img_pos', img_positions.size())
        # print('image_size', len(image_features))
        # print('image_embed', image_feature_broadcasted.size())
        # print('input_id_size', input_ids.size())
        # image_embedding = image_features.mean(dim=0, keepdim=True)  # 形状: (1, 768)
        # img_token_ids = self.encode_image_as_tokens(image_embedding)
        
        retrieved_inputs_t5, retrieved_atts_t5 = self.forward_query(retrieved_embeds, retrieved_atts, former_text_input_ids, former_text_input_masks, device)

        ########## prompt ##########
        prompt_embeds_t5 = self.prompt_tokens(bsz, device)
        prompt_atts_t5 = torch.ones(prompt_embeds_t5.size()[:-1], dtype=torch.long).to(device)

        ########## lm ##########
        with self.maybe_autocast():
            # print('input_ids2', input_ids.device)
            inputs_embeds = self.backbone_model.encoder.embed_tokens(input_ids)
            # print('input_egec2', inputs_embeds.device)
            # print(inputs_embeds)
            if len(image_feature_broadcasted.shape) == 3:
                inputs_embeds = inputs_embeds * (1 - img_positions) + image_feature_broadcasted * img_positions
            else:
                # print(img_positions)
                replace_mask = (img_positions == 1.0).squeeze(-1)
                num_replace_per_batch = replace_mask.sum(dim=1)  # (batch_size,)
                expected_replacements = image_feature_broadcasted.size(1)
                assert torch.all(num_replace_per_batch == expected_replacements), \
                    f"替换数量不匹配！pos 中有 {num_replace_per_batch} 个位置需要替换，但 replacement_tensor 提供 {expected_replacements} 个值"
                replace_indices = [mask.nonzero().squeeze(-1) for mask in replace_mask]  # List[Tensor], 每个 Tensor 形状 (num_replacements,)
                # print('replace_indices', replace_indices)
                # 4. 执行替换（避免 for 循环，使用高级索引）
                for i in range(inputs_embeds.size(0)):
                    # print(inputs_embeds.shape)
                    # print(image_feature_broadcasted.shape)
                    image_feature_broadcasted = image_feature_broadcasted.squeeze(2)
                    # print(image_feature_broadcasted.shape)
                    inputs_embeds[i, replace_indices[i]] = image_feature_broadcasted[i]
            # print(inputs_embeds.size())

            inputs_embeds = torch.cat([retrieved_inputs_t5, prompt_embeds_t5, inputs_embeds], dim=1)
            attention_mask = torch.cat([retrieved_atts_t5, prompt_atts_t5, attention_mask], dim=1)
            # print('input_embed', inputs_embeds.size())
            # print('inputs_embeds', inputs_embeds.device)
            # print('attention_mask', attention_mask.device)
            outputs = self.backbone_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_attention_mask=label_attention_mask,
                return_dict=True,
                output_hidden_states=False,
                labels=labels,
            )
            loss = outputs.loss

        return {
            "loss": loss,
            "logits": outputs.logits,
            }

    # 这里input跳过logits变成token
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        label_attention_mask=None,
        image_inputs=None,
        image_inputs_attention_mask=None,
        text_inputs=None,
        text_inputs_attention_mask=None,
        former_text_input_ids=None,
        former_text_input_masks=None,
        cold_start_p=0,
        origin_image_feature=None,
        **generate_kwargs,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        device = input_ids.device
        bsz = input_ids.shape[0]
        ########## image ##########
        image_features = origin_image_feature
        image_embeds, image_atts = self.forward_image(image_inputs, image_inputs_attention_mask, device)

        ########## text ##########
        text_embeds, text_atts = self.forward_text(text_inputs, text_inputs_attention_mask, device)

        ########## query ##########
        retrieved_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        retrieved_atts = torch.cat([image_atts, text_atts], dim=1)

        retrieved_inputs_t5, retrieved_atts_t5 = self.forward_query(retrieved_embeds, retrieved_atts, former_text_input_ids, former_text_input_masks, device)

        ########## prompt ##########
        prompt_embeds_t5 = self.prompt_tokens(bsz, device)
        prompt_atts_t5 = torch.ones(prompt_embeds_t5.size()[:-1], dtype=torch.long).to(device)


        img_token_id = self.tokenizer.convert_tokens_to_ids("[IMG]")
        img_positions = (input_ids == img_token_id).unsqueeze(-1).float()
        
        pooled_image_features = [feat.mean(dim=1).squeeze(0) for feat in image_features]

        # 转成一个 tensor，变成 (batch_size, 768)
        image_feature_tensor = torch.stack(pooled_image_features, dim=0)
        image_feature_broadcasted = image_feature_tensor.unsqueeze(1)

        # print(retrieved_inputs_t5[:3])

        ########## lm ##########
        with self.maybe_autocast():
            inputs_embeds = self.backbone_model.encoder.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds * (1 - img_positions) + image_feature_broadcasted * img_positions

            inputs_embeds = torch.cat([retrieved_inputs_t5, prompt_embeds_t5, inputs_embeds], dim=1)
            attention_mask = torch.cat([retrieved_atts_t5, prompt_atts_t5, attention_mask], dim=1)

            outputs = self.backbone_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
            # outputs = self.t5_tokenizer.batch_decode(
            #     outputs, skip_special_tokens=True
            # )

        return outputs