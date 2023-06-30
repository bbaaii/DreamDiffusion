import torch
import torch.nn as nn
from functools import partial
from PIL import Image
# import clip
import sys
sys.path.append('../dreamdiffusion/code/')
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel, AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection
from dc_ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
import kornia

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()



    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]#.to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained(version)
        self.transformer = CLIPVisionModelWithProjection.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()



    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.transformer(**inputs)
        image_embeds = outputs.image_embeds
        return image_embeds
        # z = outputs.last_hidden_state

        # return z

    def encode(self, inputs):
        return self(inputs)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model = 'ViT-L/14',
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    # from dc_ldm.util import count_params
    # text_model = FrozenCLIPEmbedder()
    # text = ['a dog']
    # text_out = text_model(text)
    # print(text_out.shape)
    # FrozenCLIPEmbedder

#     def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
#         return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


#     def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#         caption_loss = contrastive_loss(similarity)
#         image_loss = contrastive_loss(similarity.t())
#         return (caption_loss + image_loss) / 2.0

#     input = Image.open('../dreamdiffusion/datasets/imageNet_images/n02106662/n02106662_1451.JPEG')

#     from transformers import AutoProcessor, CLIPModel

#     model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#     processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)

#     inputs = processor(
#         text=["a photo of a cat", "a photo of a dog"], images=input, return_tensors="pt", padding=True
#     )
    def contrastive_loss(logits, dim):
        m = nn.functional.log_softmax(logits, dim=dim)
        print(m)
        neg_ce = torch.diag(m)
        print(neg_ce)
        print(-neg_ce.mean())
        return -neg_ce.mean()
    
    def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = contrastive_loss(similarity, dim=0)
        image_loss = contrastive_loss(similarity, dim=1)
        return (caption_loss + image_loss) / 2.0
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#     probs = logits_per_image.softmax(dim=1)  
#     print(probs)
#     print(outputs.text_embeds.shape)
#     print(outputs.image_embeds.shape)
#     f = torch.cosine_similarity(outputs.text_embeds, outputs.image_embeds, dim=-1)
#     print(f)
#     print(model.logit_scale.exp())
# # logits_per_text
#     logits_per_text = torch.matmul(outputs.text_embeds, outputs.image_embeds.t()) * model.logit_scale.exp()
#     logits_per_image = logits_per_text.t()
#     print(logits_per_text)
#     print(logits_per_image)
#     print(clip_loss(logits_per_text))
    z_i = torch.randn(4, 768)
    z_j = z_i
    # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
    # print(representations.shape)
    # print(representations.unsqueeze(1).shape)
    # print(representations.unsqueeze(0).shape)
    similarity_matrix = nn.functional.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
    print(similarity_matrix)
    print(clip_loss(similarity_matrix))
    
    # model = FrozenImageEmbedder()
    # # out = model(input)
    # # print(out.shape)

    # # model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")



    # # input = Image.open(requests.get(url, stream=True).raw)

    # inputs = processor(images=input, return_tensors="pt")
    # # for k, v in inputs.items():
    # #     print(k)
    # #     print(v.shape)
    # # print()
    # # print(inputs)

    # outputs = model(inputs)
    # # image_embeds = outputs.image_embeds
    # print(outputs.shape)


    # from transformers import AutoTokenizer, CLIPTextModelWithProjection

    # model_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # inputs_text = tokenizer(["a dog"], padding=True, return_tensors="pt")

    # outputs_text = model_text(**inputs_text)
    # text_embeds = outputs_text.text_embeds
    # f = torch.cosine_similarity(outputs, text_embeds, dim=-1)
    # print(f)

    # image_embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    #     # cosine similarity as logits
    # logit_scale = torch.tensor([2.6592]).exp()
    # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    # print(logits_per_text)
    # logits_per_image = logits_per_text.t()
    # print(logits_per_image)



    # print(outputs)
    # count_params(model, verbose=True)