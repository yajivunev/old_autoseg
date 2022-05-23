import json
from unet import conv_pass,unet
import tensorflow as tf

def create_network(input_shape, name):

    tf.reset_default_graph()

    #with tf.variable_scope('mtlsd'):

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    net, _, _ = unet(
        fmaps_in=raw_batched,
        num_fmaps=12,
        fmap_inc_factors=5,
        downsample_factors=[[1,2,2],[1,2,2],[2,2,2]],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]], [[3, 3, 3], [3, 3, 3]], [[3, 3, 3], [3, 3, 3]], [[1, 3, 3], [1, 3, 3]]],
        kernel_size_up=[[[1, 3, 3], [1, 3, 3]], [[3, 3, 3], [3, 3, 3]], [[3, 3, 3], [3, 3, 3]]],
        num_fmaps_out=12)

    affs_batched, _ = conv_pass(
        net,
        kernel_sizes=[1],
        num_fmaps=3,
        activation='sigmoid',
        name='affs')
    affs = tf.squeeze(affs_batched, axis=0)

    output_shape = tuple(affs.get_shape().as_list()[1:])

    gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
    loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

    loss = tf.losses.mean_squared_error(
        gt_affs,
        affs,
        loss_weights_affs)

    summary = tf.summary.merge([
        tf.summary.scalar('loss', loss),
    ])

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    print("input shape : %s"%(input_shape,))
    print("output shape: %s"%(output_shape,))

    tf.train.export_meta_graph(filename=name + '.meta')

    config = {
        'raw': raw.name,
        'affs': affs.name,
        'gt_affs': gt_affs.name,
        'loss_weights_affs': loss_weights_affs.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'summary': summary.name
    }

    config['outputs'] = {
            'affs':
            {"out_dims": 3,
                "out_dtype": "uint8"
                }
            }

    with open(name + '.json', 'w') as f:
        json.dump(config, f)


if __name__ == "__main__":

    create_network((48, 196, 196), 'train_net')
    #create_network((268, 268, 268), 'config')