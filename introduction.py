import tensorflow as tf
import numpy as np


def create_data(weight, bias, feat_num, sample_num):
    """
    weight:响应变量相较特征的权值
    bias:响应变量相较特征的偏差
    feat_num:特征个数
    sample_num:样本数量
    """
    assert len(weight)==feat_num, "输入列数和权值长度不一致"
    x_data = np.float32(np.random.rand(feat_num, sample_num)) # 随机输入
    y_data = np.dot(weight, x_data) + bias
    return x_data, y_data

def create_liner_model(b_init_value, W_init_value, x_data, y_data, learning_rate=0.5):
    #构造线性模型
    b = tf.Variable(b_init_value)
    W = tf.Variable(W_init_value)
    y = tf.matmul(W, x_data) + b

    #最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    return b, W, train

def run_model(iter_num, batch_num, train, W, b):
    #初始化变量
    init = tf.global_variables_initializer()
    #启动图（graph)
    sess = tf.Session()
    sess.run(init)

    #拟合平面
    for step in range(iter_num):
        sess.run(train)
        if step % batch_num == 0:
            print(step, sess.run(W), sess.run(b))



def main():
  #使用Numpy 生成假数据（phony data)， 总共100个点
  weight = [0.100, 0.200, 0.300]
  bias = 0.4000
  feat_num = 3
  sample_num = 1000
  x_data, y_data = create_data(weight, bias, feat_num, sample_num)

  #构造线性模型
  b_init_value = tf.zeros([1])
  W_init_value = tf.random_uniform([1, 3], -1.0, 1.0)
  b, W, train = create_liner_model(b_init_value, W_init_value, x_data, y_data)
  
  #初始化变量，启动图，拟合平面
  iter_num = 401
  batch_num = 50
  run_model(iter_num, batch_num, train, W, b)



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()



