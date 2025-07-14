import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import functools
import tensorflow as tf
import scipy
import sys

sys.path.append('/rds/project/rds-a1NGKrlJtrw/LLPS/')  #directory of alphafold code
from alphafold.model import mapping
from alphafold.model import prng
from alphafold.model import utils
from alphafold.model import layer_stack
from jax import lax

def glorot_uniform():
  return hk.initializers.VarianceScaling(scale=1.0,
                                         mode='fan_avg',
                                         distribution='uniform')

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)
def mean_absolute_error(rets,labels):
  loss =  jnp.mean(jnp.absolute(labels-rets),axis=-1)
  return jnp.asarray(loss)
def sigmoid_cross_entropy(logits, labels):
  """Computes sigmoid cross entropy given logits and multiple class labels."""
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  loss = -labels * log_p - (1. - labels) * log_not_p
  return jnp.asarray(loss)

def calculate_r_square(y_true,y_pred):
  return 1- np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)

def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
  """Applies dropout to a tensor."""
  if is_training and rate != 0.0:
    shape = list(tensor.shape)
    if broadcast_dim is not None:
      shape[broadcast_dim] = 1
    keep_rate = 1.0 - rate
    random_int=jnp.ravel(tensor)[0]
    int_repr = lax.bitcast_convert_type(random_int, jnp.int32)
    rng=jax.random.fold_in(safe_key.get(),int_repr)
    keep = jax.random.bernoulli(rng, keep_rate, shape=shape)
    return keep * tensor / keep_rate
  else:
    return tensor

def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    global_config,
                    output_act=None,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, mask, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

  if module.config.shared_dropout:
      broadcast_dim = 0
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act

def dropout_wrapper_pair(module,
                    input_single,
                    input_pair,
                    mask_single,
                    safe_key,
                    global_config,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""


  gc = global_config
  residual_single,residual_pair = module(input_single, mask_single, input_pair, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate
  _, *safe_subkey = safe_key.split(3)
  safe_subkey = iter(safe_subkey)
  if module.config.shared_dropout:
      broadcast_dim = 0
  else:
    broadcast_dim = None

  residual_single = apply_dropout(tensor=residual_single,
                           safe_key=next(safe_subkey),
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)
  residual_pair = apply_dropout(tensor=residual_pair,
                           safe_key=next(safe_subkey),
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)
  new_single = input_single + residual_single
  new_pair = input_pair + residual_pair

  return new_single,new_pair


def _calculate_value_from_logits(logits: np.ndarray,breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    logits: [batch_num,num_bins] unnormalized log probabilities
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    value: [batch_num], the predicted values.
  """
  step = (breaks[1] - breaks[0])
  bin_centers = breaks + step / 2
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]],axis=0)
  probs = scipy.special.softmax(logits, axis=-1)
  predicted_value = np.sum(probs * bin_centers, axis=-1)
  return predicted_value


class Linear(hk.Module):
  """Protein folding specific Linear Module.

  This differs from the standard Haiku Linear in a few ways:
    * It supports inputs of arbitrary rank
    * Initializers are specified by strings
  """

  def __init__(self,
               num_output: int,
               initializer: str = 'linear',
               use_bias: bool = True,
               bias_init: float = 0.,
               name: str = 'linear'):
    """Constructs Linear Module.

    Args:
      num_output: number of output channels.
      initializer: What initializer to use, should be one of {'linear', 'relu',
        'zeros'}
      use_bias: Whether to include trainable bias
      bias_init: Value used to initialize bias.
      name: name of module, used for name scopes.
    """

    super().__init__(name=name)
    self.num_output = num_output
    self.initializer = initializer
    self.use_bias = use_bias
    self.bias_init = bias_init

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Connects Module.

    Args:
      inputs: Tensor of shape [..., num_channel]

    Returns:
      output of shape [..., num_output]
    """
    n_channels = int(inputs.shape[-1])

    weight_shape = [n_channels, self.num_output]
    if self.initializer == 'linear':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=1.)
    elif self.initializer == 'relu':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=2.)
    elif self.initializer == 'zeros':
      weight_init = hk.initializers.Constant(0.0)

    weights = hk.get_parameter('weights', weight_shape, inputs.dtype,
                               weight_init)

    # this is equivalent to einsum('...c,cd->...d', inputs, weights)
    # but turns out to be slightly faster
    inputs = jnp.swapaxes(inputs, -1, -2)
    output = jnp.einsum('...cb,cd->...db', inputs, weights)
    output = jnp.swapaxes(output, -1, -2)

    if self.use_bias:
      bias = hk.get_parameter('bias', [self.num_output], inputs.dtype,
                              hk.initializers.Constant(self.bias_init))
      output += bias

    return output

class Transition(hk.Module):
  """Transition layer.

  Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
  Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
  """

  def __init__(self, config, global_config, name='transition_block'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask, is_training=True, output_dim=None, with_norm=True):
    """Builds Transition module.

    Arguments:
      act: A tensor of queries of size [batch_size, N_res, N_channel].
      mask: A tensor denoting the mask of size [batch_size, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      A float32 tensor of size [batch_size, N_res, output_dim].
    """
    nc = act.shape[-1]
    if output_dim is None:
      output_dim=nc

    num_intermediate = int(nc * self.config.num_intermediate_factor)
    mask = jnp.expand_dims(mask, axis=-1)

    if with_norm:
      act = hk.LayerNorm(
          axis=[-1],
          create_scale=True,
          create_offset=True,
          name='input_layer_norm')(
              act)

    transition_module = hk.Sequential([
        Linear(
            num_intermediate,
            initializer='relu',
            name='transition1'), jax.nn.relu,
        Linear(
            output_dim,
            initializer=utils.final_init(self.global_config),
            name='transition2')
    ])

    act = transition_module(act)
    return act

class mAttention(hk.Module):
  def __init__(self, config, global_config, output_dim, name='mAttention'):
      super().__init__(name=name)
      self.config=config
      self.global_config=global_config
      self.output_dim=output_dim
  def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head
    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        init=glorot_uniform())
    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    logits_update= jnp.einsum('bqhc,bkhc->bhqk', q, k)
    logits = logits_update + bias
    if nonbatched_bias is not None:
      logits += jnp.expand_dims(nonbatched_bias, axis=0)
    weights = jax.nn.softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                              gating_weights) + gating_bias

      gate_values = jax.nn.sigmoid(gate_values)

      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,),
                              init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output,logits_update

class AttentionWithPairBias(hk.Module):
  """MSA per-row attention biased by the pair representation.

  Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
  """

  def __init__(self, config, global_config,
               name='msa_row_attention_with_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
              msa_act,
              msa_mask,
              pair_act,
              is_training=False):
    """Builds MSARowAttentionWithPairBias module.

    Arguments:
      msa_act: [batch, N_res, c_m] MSA representation.
      msa_mask: [batch, N_res] mask of non-padded regions.
      pair_act: [batch, N_res, N_res, c_z] pair representation.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_res, c_m].
    """
    c = self.config
    gc= self.global_config
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert len(pair_act.shape) == 4
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = hk.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    pair_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='feat_2d_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        init=hk.initializers.RandomNormal(stddev=init_factor))
    bias += jnp.einsum('bqkc,ch->bhqk', pair_act, weights)

    attn_mod = mAttention(
        c, self.global_config, msa_act.shape[-1])
    msa_act,logits = attn_mod(msa_act, msa_act, bias)
    logits=jnp.swapaxes(logits, -2, -3)  #bhqk->bqhk
    logits=jnp.swapaxes(logits, -1, -2)  #bqhk->bqkh
    logits = Linear(
        pair_act.shape[-1],
        initializer='relu',
        name='feat_2d_update_0')(
            logits)
    logits = jax.nn.relu(logits)
    pair_act = Linear(
        pair_act.shape[-1],
        initializer=utils.final_init(gc),
        name='feat_2d_update_1')(
            logits)

    return msa_act,pair_act

class SequenceConcatenate(hk.Module):
  def __init__(self, config, global_config,
               name='sequence_concatenate'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  def __call__(self,
              act,
              act_mask,
              sequence,
              is_training=False,
              safe_key=None,
              ):
    c=self.config
    origin_channels=act.shape[-1]
    sequence=Linear(c.num_channels,name='embedding',use_bias=False)(sequence)
    act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='input_layer_norm')(
            act)
    act=jnp.concatenate((act,sequence),axis=-1)
    act=Linear(origin_channels,name='embedding')(act)
    transition_module = hk.Sequential([
        Linear(
            origin_channels*c.num_intermediate_factor,
            initializer='relu',
            name='transition1'), jax.nn.relu,
        Linear(
            origin_channels,
            initializer=utils.final_init(self.global_config),
            name='transition2')
    ])
    act = mapping.inference_subbatch(
        transition_module,
        self.global_config.subbatch_size,
        batched_args=[act],
        nonbatched_args=[],
        low_memory=not is_training)
    return act
  
class ProteinEncoder(hk.Module):
  def __init__(self, config, global_config,
               name='protein_encoder'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
              msa_act,
              struc_act,
              msa_mask,
              pair_act,
              pair_mask,
              safe_key=None,
              is_training=False):

    c = self.config
    gc=self.global_config
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert len(pair_act.shape) == 4
    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=self.global_config)
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    safe_key, *safe_subkey = safe_key.split(5)
    safe_subkey = iter(safe_subkey)
    msa_act =  Transition(c['single_transition'], gc, name='single_transition_input')(
          msa_act,
          msa_mask,
          is_training=is_training, with_norm=False)
    struc_act =  Transition(c['single_transition'], gc, name='struc_transition_input')(
          struc_act,
          msa_mask,
          is_training=is_training, with_norm=False
          )
    pair_act = Transition(c['pair_transition'], gc, name='pair_transition_input')(
          pair_act,
          pair_mask,
          is_training=is_training, with_norm=False)
    struc_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_struc')(
            struc_act)
    msa_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_msa')(
            msa_act)
    msa_act=jnp.concatenate((msa_act,struc_act),axis=-1)
    msa_act = Linear(
        c.single_channels*4,
        initializer='relu',
        name='embedding_msa1')(
            msa_act)
    msa_act=jax.nn.relu(msa_act) 
    msa_act = Linear(
        c.single_channels,
        initializer=utils.final_init(gc),
        name='embedding_msa2')(
            msa_act)
    pair_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_pair')(
            pair_act)
    pair_act = Linear(
        c.pair_channels*4,
        initializer='relu',
        name='embedding_pair1')(
            pair_act)
    pair_act=jax.nn.relu(pair_act) 
    pair_act = Linear(
        c.pair_channels-1,
        initializer=utils.final_init(gc),
        name='embedding_pair2')(
            pair_act)
    return msa_act,pair_act

class SASAHead(hk.Module):
  """Head to predict the per-residue SASA ."""


  def __init__(self, config, global_config, name='predicted_SASA_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask, is_training):

    act = Linear(
        self.config.num_channels,
        initializer='relu',
        name='act_0')(
            act)
    act = jax.nn.relu(act)
    act = Linear(
        self.config.num_channels,
        initializer='relu',
        name='act_2')(
            act)
    act = jax.nn.relu(act)
    ret = Linear(
        1,
        initializer=utils.final_init(self.global_config),
        name='logits')(
            act)
    ret=jnp.squeeze(ret,axis=-1)
    return dict(ret=ret)

  def loss(self, value, feat):
    msa_mask = feat['msa_mask']
    SASA=feat['SASA']
    rets=value['ret']
    errors=jnp.square(SASA-rets)
    loss = jnp.sum(errors * msa_mask,axis=-1) / (jnp.sum(msa_mask,axis=-1) + 1e-8)
    return loss

class SolubilityHead(hk.Module):
  def __init__(self,config, global_config, name='solubility_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  def __call__(self,act,msa_mask,pair_act,sequence,resi_num,safe_key,is_training):
    c=self.config
    gc=self.global_config
    batch_size=act.shape[0]
    num_channels=c.num_channels
    num_bins=c['num_bins']

    safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)

    act = Linear(
        num_channels,
        initializer='relu',
        name='act_0')(
            act)
    act = jax.nn.relu(act)
    act = Linear(
        num_channels,
        initializer=utils.final_init(gc),
        name='act_1')(
            act)
    act = jnp.einsum('bqc,bq->bqc',act,msa_mask)
    profile=act
    act = jnp.einsum('bqc,b->bc',act,1/resi_num)
    nn_act_2=Linear(
        num_channels*2,
        initializer='relu',
        name='act_2')

    solubility_module = hk.Sequential([
        nn_act_2,
        jax.nn.relu,
        Linear(
          num_channels,
          initializer='relu',
          name='act_sol_1'),
        jax.nn.relu,
        Linear(
        1,
        initializer=utils.final_init(gc),
        name='sol_output')
    ])

    logit = solubility_module(act)
    #sigmoid
    solubility = jax.nn.sigmoid(logit)
    if not is_training:
      sol_approx_profile=solubility_module(profile)
      return dict(solubility=solubility,sol_approx_profile=sol_approx_profile,logit=logit)
    else:
      return dict(solubility=solubility,logit=logit,)

  def loss(self,value,feat):
    solubility=jnp.expand_dims(feat['solubility'],axis=-1)
    rets_solubility=value['logit']
    loss = sigmoid_cross_entropy(rets_solubility,solubility)
    return loss

class myModel(hk.Module):
  def __init__(self, config, name='myModel'):
    super().__init__(name=name)
    self.config = config.model
    self.global_config = config.global_config
  def __call__(
      self,
      feat,
      is_training,
      compute_loss=False):
    c=self.config
    gc=self.global_config
    ret={}
    msa_act=feat['msa_act']
    msa_mask=feat['msa_mask']
    pair_act=feat['pair_act']
    pair_mask=feat['pair_mask']
    struc_act=feat['struc_act']
    resi_num=feat['resi_num']
    sequence=feat['sequence']

    msa_act_2=feat['msa_act_2']
    pair_act_2=feat['pair_act_2']
    resi_num_2=feat['resi_num_2']
    sequence_2=feat['sequence_2']
    msa_mask_2=feat['msa_mask_2']
    pair_mask_2=feat['pair_mask_2']
    struc_act_2=feat['struc_act_2']
    crop_size=msa_act.shape[-2]
    batch_size=msa_act.shape[0]

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=self.global_config)
    dropout_wrapper_pair_fn = functools.partial(
        dropout_wrapper_pair,
        is_training=is_training,
        global_config=self.global_config)

    safe_key = prng.SafeKey(hk.next_rng_key())
    safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)

    msa_act,pair_act=ProteinEncoder(c.protein_encoder,gc)(msa_act,struc_act,msa_mask,pair_act,pair_mask,
                           safe_key=next(sub_keys),is_training=is_training)
    msa_act_2,pair_act_2=ProteinEncoder(c.protein_encoder,gc)(msa_act_2,struc_act_2,msa_mask_2,pair_act_2,pair_mask_2,
                           safe_key=next(sub_keys),is_training=is_training)
    
    msa_act=jnp.concatenate((msa_act,msa_act_2),axis=-2)
    sequence=jnp.concatenate((sequence,sequence_2),axis=-2)
    pair_act=jnp.pad(pair_act,((0,0),(0,0),(0,crop_size),(0,0)),'constant',constant_values=(0.0,0.0))
    pair_act_2=jnp.pad(pair_act_2,((0,0),(0,0),(crop_size,0),(0,0)),'constant',constant_values=(0.0,0.0))
    pair_act=jnp.concatenate((pair_act,pair_act_2),axis=-3)
    

    pair_mask=jnp.concatenate(
                        (jnp.concatenate((pair_mask,jnp.einsum('bi,bj->bij',msa_mask,msa_mask_2)),axis=-1),
                         jnp.concatenate((jnp.einsum('bi,bj->bij',msa_mask_2,msa_mask),pair_mask_2),axis=-1)),
                         axis=-2)
    msa_mask=jnp.concatenate((msa_mask,msa_mask_2),axis=-1)

    seq_pos=jnp.pad(jnp.ones([batch_size,crop_size,crop_size]),((0,0),(0,0),(0,crop_size)),'constant',constant_values=(0.0,0.0))
    seq_pos=jnp.concatenate([
                          seq_pos,
                          jnp.pad(jnp.ones([batch_size,crop_size,crop_size]),((0,0),(0,0),(crop_size,0)),'constant',constant_values=(0.0,0.0))],
                          axis=-2)
    pair_act=jnp.concatenate([pair_act,jnp.expand_dims(seq_pos,axis=-1)],axis=-1)
    resi_num+=resi_num_2

    impl_1=AttentionWithPairBias(c['attention_with_pair_bias'], gc,name='impl_1')
    impl_2=Transition(c['single_transition'], gc, name='impl_2')
    impl_3=Transition(c['pair_transition'], gc, name='impl_3')

    def iteration_fn(x):
      msa_act, pair_act, safe_key = x
      safe_key, *safe_subkey = safe_key.split(4)
      safe_subkey = iter(safe_subkey)
      msa_act,pair_act=dropout_wrapper_pair_fn(
          impl_1,
          msa_act,
          pair_act,
          msa_mask,
          is_training=is_training,
          safe_key=next(safe_subkey))
      msa_act = dropout_wrapper_fn(
          impl_2,
          msa_act,
          msa_mask,
          safe_key=next(sub_keys))
      pair_act=dropout_wrapper_fn(
          impl_3,
          pair_act,
          pair_mask,
          safe_key=next(safe_subkey))
      return (msa_act, pair_act, safe_key)
      
    if gc.use_remat:
      iteration_fn = hk.remat(iteration_fn)

    iteration_stack = layer_stack.layer_stack(c['iteration_num_block'])(
        iteration_fn)
    msa_act, pair_act, safe_key = iteration_stack(
        (msa_act, pair_act, safe_key))

    #SASA_head=SASAHead(c['SASA_head'],gc,name='SASA_head')
    solubility_head=SolubilityHead(c['solubility_head'],gc,name='solubility_head')
    ret['solubility']=solubility_head(msa_act,msa_mask,pair_act,sequence,resi_num,safe_key,is_training)
    #ret['SASA']=SASA_head(msa_act,msa_mask,is_training)    #ret['test']=msa_act0
    if not compute_loss:
      return ret
    else:
      loss=solubility_head.loss(ret['solubility'],feat)
      #loss+=SASA_head.loss(ret['SASA'],feat)*SASA_head.config.weight
      return ret,loss
