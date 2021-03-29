import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
	def __init__(self, k, heads=8):
		super(SelfAttention, self).__init__()
		self.k, self.heads = k, heads

		# these compute the queries, keys, values for all
		# heads (as a single concatenated vector)
		self.tokeys = nn.Linear(k, k * heads, bias=False)
		self.toqueries = nn.Linear(k, k * heads, bias=False)
		self.tovalues = nn.Linear(k, k * heads, bias=False)

		# this unifies the outputs of the different heads into a single k-vector
		self.unifyheads = nn.Linear(heads * k, k)

	def forward(self, x):
		b, t, k = x.size()
		h = self.heads

		queries = self.toqueries(x).view(b, t, h, k)
		keys = self.tokeys(x).view(b, t, h, k)
		values = self.values(x).view(b, t, h, k)
		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
		values = values.transpose(1, 2).contiguous().view(b * h, t, k)

		queries = queries / (k ** (1/4))
		keys = keys / (k ** (1/4))

		# - get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))
		# - dot has size (b*h, t, t) containing raw weights

		dot = F.softmax(dot, dim=2)
		# - dot now contains row-wise normalized weights

		# apply the self attention to the values
		out = torch.bmm(dot, values).view(b, h, t, k)

		# - swap h, t back  and unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, h * k)
		return self.unifyheads(out)

class TransformerBlock(nn.Module):
	def __init__(self, k, heads):
		super().__init__()

		self.attention = SelfAttention(k, heads=heads)

		self.norm1 = nn.LayerNorm(k)
		self.norm2 = nn.LayerNorm(k)

		self.ff = nn.Sequential(
			nn.Linear(k, 4*k),
			nn.ReLU(),
			nn.Linear(4*k, k)
		)

	def forward(self, x):
		attended = self.attention(x)
		x = self.norm1(attended + x)

		fedforward = self.ff(x)
		return self.norm2(fedforward + x)


class Transformer(nn.Module):
	def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
		super().__init__()

		self.num_tokens = num_tokens
		self.token_emb = nn.Embedding(num_tokens, k)
		self.pos_emb = nn.Embedding(seq_length, k)

		# the sequence of transformer blocks that does all the heavy lifting
		tblocks = []
		for i in range(depth):
			tblocks.append(TransformerBlock(k=k, heads=heads))
		self.tblocks = nn.Sequential(*tblocks)

		# maps the final output sequence to class logits
		self.toprobs = nn.Linear(k, num_classes)

	def forward(self, x):
		"""
		:param x: A (b,t) tensor of integer values representing words 
				(in some predetermined vocabulary)
		:return: A (b, c) tensor of log-probabilities over the classes
				(where c is the nr. of classes)
		"""

		# generate token embeddings
		tokens = self.token_emb()
		b, t, k = tokens.size()

		# generate position embeddings
		positions = torch.arange(t)
		positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

		x = tokens + positions
		x = self.tblocks(x)

		# average-pool over the t dimension and project to class probabilities
		x = self.toprobs(x.mean(dim=1))
		return F.log_softmax(x, dim=1)