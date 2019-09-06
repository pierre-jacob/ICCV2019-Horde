
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer


class RandomMaclaurin(Initializer):
    """
    Random Maclaurin initializer.
    See Yang Gao, Oscar Beijbom, Ning Zhang, Trevor Darrell. Compact Bilinear Pooling. In IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR). June 2016.
    http://openaccess.thecvf.com/content_cvpr_2016/papers/Gao_Compact_Bilinear_Pooling_CVPR_2016_paper.pdf
    """
    def __call__(self, shape, dtype=None, partition_info=None):
        matrix = np.random.choice([-1., +1.], shape)
        return tf.convert_to_tensor(matrix, dtype=dtype)
