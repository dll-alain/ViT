"""
refer to:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np


class PatchEmbed(layers.Layer):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  kernel_initializer=initializers.LecunNormal,
                                  bias_initializer=initializers.Zeros())

    def call(self, inputs, **kwargs):
        # B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        B, H, W, C = inputs.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(inputs)
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [B, self.num_patches, self.embed_dim])
        return x


class ConcatClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=768, num_patches=196, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def build(self, input_shape):
        self.cls_token = self.add_weight(name="cls",
                                         shape=[1, 1, self.embed_dim],
                                         initializer=initializers.Zeros(),
                                         trainable=True,
                                         dtype=tf.float32)
        self.pos_embed = self.add_weight(name="pos_embed",
                                         shape=[1, self.num_patches + 1, self.embed_dim],
                                         initializer=initializers.RandomNormal(stddev=0.02),
                                         trainable=True,
                                         dtype=tf.float32)

    def call(self, inputs, **kwargs):
        batch_size, _, _ = inputs.shape

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)  # [B, 197, 768]
        x = x + self.pos_embed

        return x


class Attention(layers.Layer):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.Zeros()

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None):
        super(Attention, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name="out",
                                 kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = inputs.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(inputs)
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        # transpose: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class MLP(layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(in_features, name="Dense_1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class Block(layers.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 name=None):
        super(Block, self).__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              name="MultiHeadAttention")
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) if drop_path_ratio > 0. \
            else layers.Activation("linear")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = MLP(dim, drop=drop_ratio, name="MlpBlock")

    def call(self, inputs, training=None):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class VisionTransformer(Model):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 representation_size=None, num_classes=1000, prototypes=600, name="ViT-B/16"):
        super(VisionTransformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.qkv_bias = qkv_bias

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim,
                                                               num_patches=num_patches,
                                                               name="cls_pos")

        self.pos_drop = layers.Dropout(drop_ratio)

        dpr = np.linspace(0., drop_path_ratio, depth)  # stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                             drop_path_ratio=dpr[i], name="encoderblock_{}".format(i))
                       for i in range(depth)]

        self.norm = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")

        # ds
        self.ds1 = DS1(prototypes, embed_dim, name="ds1")
        self.ds1_activate = DS1_activate(prototypes, name="ds1_activate")
        self.ds2 = DS2(prototypes, num_class=num_classes, name="ds2")
        self.ds2_omega = DS2_omega(prototypes, num_class=num_classes, name="omega")
        self.ds3 = DS3_Dempster(prototypes, num_class=num_classes, name="ds3")
        self.ds_normalize = DS3_normalize()
        self.dm = DM(0.8, num_class=num_classes, name="DM")

        # if representation_size:
        #     self.has_logits = True
        #     self.pre_logits = layers.Dense(representation_size, activation="tanh", name="pre_logits")
        # else:
        #     self.has_logits = False
        #     self.pre_logits = layers.Activation("linear")
        #
        # self.head = layers.Dense(num_classes, name="head", kernel_initializer=initializers.Zeros())

    def call(self, inputs, training=None):
        # [B, H, W, C] -> [B, num_patches, embed_dim]
        x = self.patch_embed(inputs)  # [B, 196, 768]
        x = self.cls_token_pos_embed(x)  # [B, 176, 768]
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        # ds
        x = self.ds1(x[:, 0])
        x = self.ds1_activate(x)
        x = self.ds2(x)
        x = self.ds2_omega(x)
        x = self.ds3(x)
        x = self.ds_normalize(x)
        x = self.dm(x)


        # x = self.pre_logits(x[:, 0])
        # x = self.head(x)

        return x


class DS1(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, name=None):
        super(DS1, self).__init__(name=name)
        self.w = self.add_weight(
            name='Prototypes',
            shape=(units, input_dim),
            initializer='random_normal',
            trainable=True
        )

        self.units = units

    def call(self, inputs):
        for i in range(self.units):
            if i == 0:
                un_mass_i = tf.subtract(self.w[i, :], inputs, name=None)
                un_mass_i = tf.square(un_mass_i, name=None)
                un_mass_i = tf.reduce_sum(un_mass_i, -1, keepdims=True)
                un_mass = un_mass_i

            if i >= 1:
                un_mass_i = tf.subtract(self.w[i, :], inputs, name=None)
                un_mass_i = tf.square(un_mass_i, name=None)
                un_mass_i = tf.reduce_sum(un_mass_i, -1, keepdims=True)
                un_mass = tf.concat([un_mass, un_mass_i], -1)
        return un_mass


class DS1_activate(tf.keras.layers.Layer):
    def __init__(self, input_dim, name=None):
        super(DS1_activate, self).__init__(name=name)
        self.xi = self.add_weight(
            name='xi',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )

        self.eta = self.add_weight(
            name='eta',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )

        self.input_dim = input_dim

    def call(self, inputs):
        gamma = tf.square(self.eta, name=None)
        alpha = tf.negative(self.xi, name=None)
        alpha = tf.exp(alpha, name=None) + 1
        alpha = tf.divide(1, alpha, name=None)
        si = tf.multiply(gamma, inputs, name=None)
        si = tf.negative(si, name=None)
        si = tf.exp(si, name=None)
        si = tf.multiply(si, alpha, name=None)
        si = si / (tf.reduce_max(si, axis=-1, keepdims=True) + 0.0001)
        return si


class DS2(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, name=None):
        super(DS2, self).__init__(name=name)
        self.beta = self.add_weight(
            name='beta',
            shape=(input_dim, num_class),
            initializer='random_normal',
            trainable=True
        )

        self.input_dim = input_dim
        self.num_class = num_class

    def call(self, inputs):
        beta = tf.square(self.beta, name=None)
        beta_sum = tf.reduce_sum(beta, -1, keepdims=True)
        u = tf.divide(beta, beta_sum, name=None)
        inputs_new = tf.expand_dims(inputs, -1)
        for i in range(self.input_dim):
            if i == 0:
                mass_prototype_i = tf.multiply(u[i, :], inputs_new[:, i], name=None)
                mass_prototype = tf.expand_dims(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = tf.expand_dims(tf.multiply(u[i, :], inputs_new[:, i], name=None), -2)
                mass_prototype = tf.concat([mass_prototype, mass_prototype_i], -2)
        mass_prototype = tf.convert_to_tensor(mass_prototype)
        return mass_prototype


class DS2_omega(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, name=None):
        super(DS2_omega, self).__init__(name=name)
        self.input_dim = input_dim
        self.num_class = num_class

    def call(self, inputs):
        mass_omega_sum = tf.reduce_sum(inputs, -1, keepdims=True)
        mass_omega_sum = tf.subtract(1., mass_omega_sum[:, :, 0], name=None)
        mass_omega_sum = tf.expand_dims(mass_omega_sum, -1)
        mass_with_omega = tf.concat([inputs, mass_omega_sum], -1)
        return mass_with_omega


class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, name=None):
        super(DS3_Dempster, self).__init__(name=name)
        self.input_dim = input_dim
        self.num_class = num_class

    def call(self, inputs):
        m1 = inputs[:, 0, :]
        omega1 = tf.expand_dims(inputs[:, 0, -1], -1)
        for i in range(self.input_dim - 1):
            m2 = inputs[:, (i + 1), :]
            omega2 = tf.expand_dims(inputs[:, (i + 1), -1], -1)
            combine1 = tf.multiply(m1, m2, name=None)
            combine2 = tf.multiply(m1, omega2, name=None)
            combine3 = tf.multiply(omega1, m2, name=None)
            combine1_2 = tf.add(combine1, combine2, name=None)
            combine2_3 = tf.add(combine1_2, combine3, name=None)
            combine2_3 = combine2_3 / tf.reduce_sum(combine2_3, axis=-1, keepdims=True)  # 后加的
            m1 = combine2_3
            omega1 = tf.expand_dims(combine2_3[:, -1], -1)
        return m1


class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self):
        super(DS3_normalize, self).__init__()

    def call(self, inputs):
        mass_combine_normalize = inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
        return mass_combine_normalize


class DM(tf.keras.layers.Layer):
    def __init__(self, nu, num_class, name=None):
        super(DM, self).__init__(name=name)
        self.nu = nu
        self.num_class = num_class

    def call(self, inputs):
        upper = tf.expand_dims((1 - self.nu) * inputs[:, -1], -1)
        upper = tf.tile(upper, [1, self.num_class + 1])
        outputs = tf.add(inputs, upper, name=None)[:, 0:-1]
        return outputs


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              prototypes=1000,
                              name="ViT-B_16")
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-B_32")
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_16")
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_32")
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-H_14")
    return model


