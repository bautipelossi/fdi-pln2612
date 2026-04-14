from collections import Counter 

class BPETokenizer:
    
    def __init__(self,text,num_merges=500):
        chars = sorted(set(text)) #saca todos los caracteres de tu texto sin duplicar (bytes)
        self.tok2id =   #generar un diccionario donde las claves son los chars y el primer elemento el ìndice y se le asigna a cada caracter un id
        self.id2tok =  #diccionario reverso (a cada id se le devuelve el caracter)

        tokens = #tokeniza el texto 
        self.merges = #mergeo los pares mas frecuentes

        for _ in range(num_merges):
            pairs = Counter(zip(tokens,tokens[1:]))

            if not pairs:
                break
            
            new_id = len(self.tok2id)
            new_tok =



    def _apply_merge(toke,ns,a,b,new_id)
        """ Recorre todo el texto y Reemplaza todas las ocurrencias del par (a,b) por new_id"""
    
    def encode(self,text):
        """codifica texto aplicando los merges aprendidos"""
    
    def decode(self, ids)
        

# ---------------atencion.py
import torch
import torch.nn  as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, d_model, n_tokens, n_heads, n_dropout=0.1):
        self.W_qs = [nn.Linear(d_model,d_model) for _ in range(n_heads)] #pytorch llama linear a las matrices
        self.W_ks = [nn.Linear(d_model,d_model) for _ in range(n_heads)] 
        self.W_vs = [nn.Linear(d_model,d_model) for _ in range(n_heads)] 
        self.dropout = nn.Dropout(n_dropout) #modelo de pytorch con hiperparametro (porcentaje de neuronas que desactivo)
    
    def forward(self, X):
        """"
        X = tensor de los embeddings
        """

        Q = self.W_q @ X
        K = self.W_k @ X
        V = self.W_v @ X

        A = Q @ K.T
        A /= math.sqrt(self.d_model) #normalizo
        A = F.softmax(A)

        Z = A @ V

        return Z


# modelo.py
# 
class FeedForward(nn.Modulo)

class Block():
# bloque transformer pre norm: LN - Attn - residual - LN - FF - RESIDUAL       
#EN EL INIT usa nn.Layernorm

    def forward


        

