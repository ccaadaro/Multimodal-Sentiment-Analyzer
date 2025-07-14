import torch
import torch.nn as nn
import torchvision.models as models

class GatedMultimodalFusion(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, output_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim + image_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_feat, image_feat):
        combined = torch.cat([text_feat, image_feat], dim=1)
        gated = self.gate(combined) * combined
        return self.classifier(gated)


class MultimodalSentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embeddings):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = False

        self.text_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove fc
        self.image_encoder = nn.Sequential(*modules)
        self.image_fc = nn.Linear(512, hidden_dim)

        self.fusion = GatedMultimodalFusion(hidden_dim, hidden_dim, hidden_dim, output_dim)

    def forward(self, image, text):
        text_emb = self.embedding(text)
        _, h_n = self.text_encoder(text_emb)
        text_feat = h_n.squeeze(0)  # shape: (batch, hidden)

        img_feat = self.image_encoder(image).squeeze()
        if len(img_feat.shape) == 1:
            img_feat = img_feat.unsqueeze(0)
        img_feat = self.image_fc(img_feat)

        return self.fusion(text_feat, img_feat)
