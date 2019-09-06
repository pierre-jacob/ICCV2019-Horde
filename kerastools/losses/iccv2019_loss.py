
import tensorflow as tf


def make_abe_loss(n_class, n_ensemble, beta_neg=0.5, beta_div=0.5, eps=1e-12, lambda_div=0.5, name='abe_loss'):
    """Convention: im1_c1, im2_c1, im3_c1, ..., im1_c2, im2_c2, ..., im1_cM, ..., imN_cM

    Compute divergence loss + contrastive loss.
    """
    def abe_loss(y_true, y_pred):
        with tf.name_scope(name=name):
            s = tf.shape(y_pred)
            c = s[0] // n_class
            d = s[1] // n_ensemble

            # Compute distance matrix:
            embeddings = tf.reshape(y_pred, tf.stack([s[0] * n_ensemble, d]))  # N * n_ensemble x embedding_learner_size
            similarity_matrix = tf.matmul(embeddings, tf.transpose(embeddings, (1, 0)))  # N * n_ensemble x N * n_ensemble
            similarity_matrix = tf.reshape(similarity_matrix, tf.stack([n_class, c, n_ensemble, n_class, c, n_ensemble]))
            per_class_similarity_matrix = tf.transpose(similarity_matrix, tf.stack([0, 3, 2, 5, 1, 4]))  # n_class x n_class x n_ensemble x n_ensemble x c x c

            # Get indexes of similarities between queries and their respective positives:
            ind_p = tf.where(tf.equal(tf.eye(c, dtype=tf.int64), 0))

            loss_p = 0.
            loss_n = 0.
            div_loss = 0.
            n_z_p = tf.cast(n_ensemble * n_class * c * (c - 1), tf.float32)
            n_z_n = 0.
            div_n_z = 0.
            for i in range(n_class):
                for j in range(n_class):
                    for k in range(n_ensemble):
                        for l in range(n_ensemble):
                            sim = per_class_similarity_matrix[i, j, k, l]  # c x c
                            if k != l:  # Compute divergence loss between each weak learners:
                                tmp_loss = tf.nn.relu(sim - beta_div)
                                div_loss += tf.reduce_sum(tmp_loss)
                                div_n_z += tf.reduce_sum(tf.cast(tf.less(0., tmp_loss), tf.float32))
                            elif i == j and k == l:  # same ensemble and same class -> contrastive loss for positive match
                                loss_p += tf.reduce_sum(tf.square((tf.gather_nd(sim, ind_p) - 1.)))
                            elif i != j and k == l:  # same ensemble but different class -> contrastive loss for negative match
                                tmp_loss = tf.nn.relu(sim - beta_neg)
                                loss_n += tf.reduce_sum(tmp_loss)
                                n_z_n += tf.reduce_sum(tf.cast(tf.less(0., tmp_loss), tf.float32))

            # Compute the global loss
            loss = loss_p / (2. * n_z_p + eps) + loss_n / (2. * n_z_n + eps) + lambda_div * div_loss / (div_n_z + eps)

        return loss

    return abe_loss
