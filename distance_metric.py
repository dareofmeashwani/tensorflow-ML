import tensorflow as tf
def euclid_distance(vector1, vector2):
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(vector1, vector2), 2)))

def minkowski_distance(vector1, vector2, order):
    return tf.pow(tf.reduce_sum(tf.pow(tf.subtract(vector1, vector2), order)), 1 / order)

def manhattan_distance(vector1, vector2):
    return tf.reduce_sum(tf.abs(tf.subtract(vector1, vector2)))

def cosine_similarity(vector1, vector2):
    numerator = tf.reduce_sum(tf.multiply(vector1, vector2),axis=1)
    denominator = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vector1),axis=1)),
                              tf.sqrt(tf.reduce_sum(tf.square(vector2),axis=1)))
    return tf.divide(numerator, denominator)

def cityblock_distance(vector1, vector2):
    return tf.reduce_sum(tf.abs(tf.subtract(vector1, vector2)))

def mahalanobis_distance(self, vector1, vector2):
    # md = (x - y) * LA.inv(R) * (x - y).T
    return 0

def jaccard_similarity(vector1, vector2):
    intersection_cardinality = tf.sets.set_intersection(vector1, vector2)
    union_cardinality = tf.sets.set_union(vector1, vector2)
    return tf.divide(intersection_cardinality, union_cardinality)