import tensorflow as tf
from collections import namedtuple, OrderedDict,defaultdict
from copy import copy
from itertools import chain

from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import Input, Lambda


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c

DEFAULT_GROUP_NAME='DEFAULT_GROUP'

class SequencePoolingLayer(tf.keras.layers.Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list

class WeightedSequenceLayer(tf.keras.layers.Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len,seq_weight]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``
      Output shape
        - 3D tensor with shape: ``(batch_size, T, embedding_size)``.
      Arguments
        - **weight_normalization**: bool.Whether normalize the weight score before applying to sequence.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            key_input, key_length_input, value_input = input_list
            mask = tf.sequence_mask(key_length_input,
                                    self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)

        if self.weight_normalization:
            value_input = softmax(value_input, dim=1)

        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            value_input = tf.tile(value_input, [1, 1, embedding_size])

        return tf.multiply(key_input, value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self, ):
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


class Hash(tf.keras.layers.Layer):
    """Looks up keys in a table when setup `vocabulary_path`, which outputs the corresponding values.
    If `vocabulary_path` is not set, `Hash` will hash the input to [0,num_buckets). When `mask_zero` = True,
    input value `0` or `0.0` will be set to `0`, and other value will be set in range [1,num_buckets).
    The following snippet initializes a `Hash` with `vocabulary_path` file with the first column as keys and
    second column as values:
    * `1,emerson`
    * `2,lake`
    * `3,palmer`
    >>> hash = Hash(
    ...   num_buckets=3+1,
    ...   vocabulary_path=filename,
    ...   default_value=0)
    >>> hash(tf.constant('lake')).numpy()
    2
    >>> hash(tf.constant('lakeemerson')).numpy()
    0
    Args:
        num_buckets: An `int` that is >= 1. The number of buckets or the vocabulary size + 1
            when `vocabulary_path` is setup.
        mask_zero: default is False. The `Hash` value will hash input `0` or `0.0` to value `0` when
            the `mask_zero` is `True`. `mask_zero` is not used when `vocabulary_path` is setup.
        vocabulary_path: default `None`. The `CSV` text file path of the vocabulary hash, which contains
            two columns seperated by delimiter `comma`, the first column is the value and the second is
            the key. The key data type is `string`, the value data type is `int`. The path must
            be accessible from wherever `Hash` is initialized.
        default_value: default '0'. The default value if a key is missing in the table.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_buckets, mask_zero=False, vocabulary_path=None, default_value=0, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        self.vocabulary_path = vocabulary_path
        self.default_value = default_value
        if self.vocabulary_path:
            initializer = TextFileInitializer(vocabulary_path, 'string', 1, 'int64', 0, delimiter=',')
            self.hash_table = StaticHashTable(initializer, default_value=self.default_value)
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):

        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        if self.vocabulary_path:
            hash_x = self.hash_table.lookup(x)
            return hash_x

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except AttributeError:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, 'vocabulary_path': self.vocabulary_path,
                  'default_value': self.default_value}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict



def get_dense_input(features, feature_columns):
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """ Dense feature
    Args:
        name: feature name,
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name




def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list), vocabulary_path=fc.vocabulary_path)(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict



def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding




DEFAULT_GROUP_NAME = "default_group"





def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features



def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list