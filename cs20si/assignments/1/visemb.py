import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow.contrib.eager as tfe
import word2vec_utils
import os

#tfe.enable_eager_execution()

VOCAB_SIZE = 50000
EMBED_SIZE = 128

def visualize2(visual_fld, num_visualize):
    word2vec_utils.most_common_words(visual_fld, num_visualize)
    embed_matrix = tf.get_variable(
        'embed_matrix',
        shape=[VOCAB_SIZE, EMBED_SIZE],
        initializer=tf.random_uniform_initializer()
    )
    saver = tf.train.Saver([embed_matrix])
    with tf.Session() as sess:
        saver.restore(sess, './embed_saved/')
        final_embed_matrix = sess.run(embed_matrix)
        embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(visual_fld)
        
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)

def main():
    visualize2('visualization/', 100)

main()
