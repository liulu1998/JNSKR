import os
import tensorflow as tf


class JNSKR:
    def __init__(self, n_users, n_items, n_relations, n_entities, max_user_pi,
                 max_relation_pi, relation_test, tail_test, negative_c, negative_ck, args):
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities

        """
        max_user_pi 是对于每个商品，最多的交互用户的数量；(所有商品中 取得最大的，对齐)
        max_relation_pi 是对于每个商品最多的交互关系数量。
        
        因为 tf 中的输入需要是一个对齐的tensor, 
        所以我们输入的每个训练batch的shape为 [batch_size, max_user_pi]，
        对于数量不足的商品，我们用“self.n_users”补齐。
        """
        self.max_user_pi = max_user_pi
        self.max_relation_pi = max_relation_pi
        # embedding size
        self.embedding = args.embed_size

        # TODO ?
        self.negative_c = tf.constant(negative_c, dtype=tf.float32)
        self.negative_ck = tf.constant(negative_ck, dtype=tf.float32)

        # weight of multi-task
        self.coefficient = args.coefficient
        self.lambda_bilinear = args.lambda_bilinear

        self.relation_test = relation_test
        self.tail_test = tail_test
        self.attention_size = args.embed_size / 2

    def _create_placeholders(self):
        """ placeholder: 占位符, 可以初始化数据类型/形状, 也可以不初始化
        placeholder 作为 模型的输入 变量
        """
        # input_items
        self.input_i = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        # TODO: input item & user ? 输入是每个 item 与它交互的 user ?
        self.input_iu = tf.placeholder(tf.int32, [None, self.max_user_pi], name="input_iu")

        # TODO hr —— head & relation ?  每个实体 和 它邻接的 边（关系） ?
        self.input_hr = tf.placeholder(tf.int32, [None, self.max_relation_pi], name="input_hr")
        # TODO ht —— head & tail ?  每个实体 和 它邻接的 节点（尾实体）?
        self.input_ht = tf.placeholder(tf.int32, [None, self.max_relation_pi], name="input_ht")

        # dropout 比例
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_kg = tf.placeholder(tf.float32, name="dropout_kg")

        self.users = tf.placeholder(tf.int32, shape=[None, ], name='users')
        self.pos_items = tf.placeholder(tf.int32, shape=[None, ], name='pos_items')

    def _create_variables(self):
        """ Variables 都是模型的权重
        """
        # user 嵌入层
        self.uid_W = tf.Variable(tf.truncated_normal(shape=[self.n_users + 1, self.embedding], mean=0.0,
                                                     stddev=0.01), dtype=tf.float32, name="uidWg")
        # item 嵌入层
        self.iid_W = tf.Variable(tf.truncated_normal(shape=[self.n_items + 1, self.embedding], mean=0.0,
                                                     stddev=0.01), dtype=tf.float32, name="iidWg")
        # entity 嵌入层
        self.eid_W = tf.Variable(tf.truncated_normal(shape=[self.n_entities + 1, self.embedding], mean=0.0,
                                                     stddev=0.01), dtype=tf.float32, name="eidWg")
        # relation 嵌入层
        self.rid_W = tf.Variable(tf.truncated_normal(shape=[self.n_relations + 1, self.embedding], mean=0.0,
                                                     stddev=0.01), dtype=tf.float32, name="ridWg")

        # item domain
        # Equ (9) prediction vector ?
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding, 1]), name="hi")

        # >>> attention 参数
        # 转换矩阵 W
        self.WA = tf.Variable(
            tf.truncated_normal(shape=[self.embedding, self.attention_size], mean=0.0,
                                stddev=tf.sqrt(tf.div(2.0, self.attention_size + self.embedding))),
            dtype=tf.float32, name='WA'
        )
        # 偏置 bias
        self.BA = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BA")
        # 外面的一层 h_{\alpha}
        self.HA = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HA")
        # <<< attention 参数

    def _attentive_sum(self, pos_r, pos_t, pos_num_r):
        """ attention 机制, 计算 注意力权重 ! 注意力权重！ 即 注意力系数！
        不是 向量表示！ 不是向量表示！
        Args:
            pos_r:
            pos_t:
            pos_num_r:

        Returns:
        """
        # Equ (11), 与论文中稍有不同, 不过无所谓
        # HA : h_{\alpha} in Equ (11)
        # BA : b in Equ (11)
        entities_j = tf.exp(
            # for a batch of matrix, 每个 matrix 乘 HA
            tf.einsum('abc,ck->abk', tf.nn.relu(
                # for a batch of matrix, 每个 matrix 乘 WA
                tf.einsum('abc,ck->abk', pos_r * pos_t, self.WA) + self.BA
            ), self.HA)
        )
        # C_{abc} = \sigma_{a} \sigma_{b} Aab · B_{abc}
        entities_j = tf.einsum('ab, abc->abc', pos_num_r, entities_j)
        # Equ (11) 分母
        entities_sum = tf.reduce_sum(entities_j, 1, keep_dims=True)
        entities_w = tf.div(entities_j, entities_sum)
        return entities_w

    def _create_inference(self):
        """ 创建推理 ? 应该是 最核心的 方法
        """
        # item

        # iid_W: item 嵌入层
        # iid: 给定 item 的 嵌入向量
        self.iid = tf.nn.embedding_lookup(self.iid_W, self.input_i)
        self.iid = tf.reshape(self.iid, [-1, self.embedding])

        # self.c = c_{v}^{-}, 推荐系统中 负例 的权重
        # self.ck 即 w_{h}^{-}, KG 中 负例的权重
        self.c = tf.nn.embedding_lookup(self.negative_c, self.input_i)
        self.ck = tf.nn.embedding_lookup(self.negative_ck, self.input_i)

        # 本文章主要依靠 Dropout 正则化
        self.iid_kg = tf.nn.dropout(self.iid, self.dropout_kg)

        # knowledge
        # >>> KG, 计算 g_{hrt}^
        # pos_r: 关系的嵌入向量
        self.pos_r = tf.nn.embedding_lookup(self.rid_W, self.input_hr)
        # pos_t: tail 实体的嵌入向量
        self.pos_t = tf.nn.embedding_lookup(self.eid_W, self.input_ht)

        """
        input_hr 是用 n_relations 对齐过的（其中有许多没用的数值, 即 n_relations）
        pos_num_r 是过滤出真正有用的元素（即 不等于 n_relations 的元素）,
        pos_num_r 是二值的 bool 矩阵
        """
        # tf.cast 数据类型转换, 此处转为 float
        self.pos_num_r = tf.cast(tf.not_equal(self.input_hr, self.n_relations), 'float32')
        #
        self.pos_r = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_r)
        self.pos_t = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_t)

        self.pos_rt = self.pos_r * self.pos_t

        # pos_hrt :　原文中的 g_{hrt}^,  (h, r, t) 的 scoring function 值
        # Equ (6)
        self.pos_hrt = tf.einsum('ac,abc->ab', self.iid_kg, self.pos_rt)
        self.pos_hrt = tf.reshape(self.pos_hrt, [-1, self.max_relation_pi])
        # <<< KG, 计算 g_{hrt}^

        # cf 推荐系统
        # Equ (10)
        # entities_w : 注意力权重, Equ (10) 的 \alpha _{(r,t)}
        self.entities_w = self._attentive_sum(self.pos_r, self.pos_t, self.pos_num_r)

        # >>> 计算 Item 的最终表示
        # kid: e_{N_v}, Equ (10)
        self.kid = tf.reduce_sum(tf.multiply(self.entities_w, self.pos_t), 1)
        # kid_drop 是 kid 经过 Dropout
        self.kid_drop = tf.nn.dropout(self.kid, self.dropout_kg)
        # iid_cf 是 iid 经过 Dropout (iid 是 Item 的嵌入向量, 即 本身表示)
        self.iid_cf = tf.nn.dropout(self.iid, self.dropout_keep_prob)

        # Equ (10), cal q_v, Item 的最终表示
        self.iid_drop = self.iid_cf + self.kid_drop
        # <<< 计算 Item 的最终表示

        self.pos_user = tf.nn.embedding_lookup(self.uid_W, self.input_iu)

        """
        因为我们的输入tensor是补齐的，
        所以首先用 tf.cast(tf.not_equal(self.input_iu, self.n_users) 操作来所补得“self.n_users”置为0，
        这样就不会被更新计算到。
        
        input_iu 是用 n_users 对齐过的（其中有许多没用的数值, 即 n_users)
        pos_num_u 是过滤出真正有用的元素（即 不等于 n_users 的元素）,
        pos_num_u 是二值的 bool 矩阵
        """
        self.pos_num_u = tf.cast(tf.not_equal(self.input_iu, self.n_users), 'float32')
        self.pos_user = tf.einsum('ab,abc->abc', self.pos_num_u, self.pos_user)

        # pos_iu : 原文中的 y_{uv}^, 预测的 u 对 v 的偏好
        # Equ (9)
        self.pos_iu = tf.einsum('ac,abc->abc', self.iid_drop, self.pos_user)
        self.pos_iu = tf.einsum('ajk,kl->ajl', self.pos_iu, self.H_i)
        self.pos_iu = tf.reshape(self.pos_iu, [-1, self.max_user_pi])
        # predict

    def _create_loss(self):
        # >>> CF 损失
        # Equ (12) 下半部分
        # H_i 是 attention network 的参数
        # TODO: 注意, idd_drop 是 Item 的最终表示
        self.loss1 = tf.reduce_sum(tf.einsum('ab,ac->bc', self.uid_W, self.uid_W)
                                   * tf.einsum('ab,ac->bc', self.c * self.iid_drop, self.iid_drop)
                                   * tf.matmul(self.H_i, self.H_i, transpose_b=True)
                                   )
        # Equ (12) 上半部分, c_{v}^{+} = 1.0,  c_{v}^{-} = self.c
        self.loss1 += tf.reduce_sum((1.0 - self.c) * tf.square(self.pos_iu) - 2.0 * self.pos_iu)
        # <<< CF 损失

        # >>> KGE 损失
        # TODO: 注意, idd_kg 是 Item 的本身表示 (没加入 邻居 的信息)
        # Equ (8), 优化后的 L_{KG}^{A}, the loss of all data , ck 即 w_{h}^{-}
        self.loss2 = tf.reduce_sum(tf.einsum('ab,ac->bc', self.ck * self.iid_kg, self.iid_kg)
                                   * tf.einsum('ab,ac->bc', self.eid_W, self.eid_W)
                                   * tf.einsum('ab,ac->bc', self.rid_W, self.rid_W))
        # ck 是 paper 中的 w_{hrt}^{-} ?, 负样本 (h,r,t) 的权重;   w_{hrt}^{+} 是 1.0
        # Equ (5), L_{KG}^{P}, the loss for positive data
        self.loss2 += tf.reduce_sum((1.0 - self.ck) * tf.square(self.pos_hrt) - 2.0 * self.pos_hrt)
        # <<< KGE 损失

        # KGE 的 L2 loss, 4 个嵌入矩阵
        self.l2_loss_0 = tf.nn.l2_loss(self.uid_W) + tf.nn.l2_loss(self.eid_W) + \
                         tf.nn.l2_loss(self.iid_W) + tf.nn.l2_loss(self.rid_W)
        # attention 的 L2 loss
        self.l2_loss_1 = tf.nn.l2_loss(self.WA) + tf.nn.l2_loss(self.BA) + tf.nn.l2_loss(self.HA)

        # CF loss, KG loss
        self.loss1 = self.coefficient[0] * self.loss1
        self.loss2 = self.coefficient[1] * self.loss2

        self.loss = self.loss1 + self.loss2 + \
                    self.lambda_bilinear[0] * self.l2_loss_0 + \
                    self.lambda_bilinear[1] * self.l2_loss_1

    def eval(self, sess, feed_dict):
        """ 验证 evaluate
        """
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

    def _creat_prediction(self):
        """
        calls: _attention_sum
        """
        pos_r = tf.nn.embedding_lookup(self.rid_W, self.relation_test)
        pos_t = tf.nn.embedding_lookup(self.eid_W, self.tail_test)

        # 过滤掉 对齐中的没用元素（即 n_relations）
        pos_num_r = tf.cast(tf.not_equal(self.relation_test, self.n_relations), 'float32')
        pos_t = tf.einsum('ab,abc->abc', pos_num_r, pos_t)

        # 注意力权重
        entities_w = self._attentive_sum(pos_r, pos_t, pos_num_r)
        k_test = tf.reduce_sum(tf.multiply(entities_w, pos_t), 1)

        pos_i_e = tf.nn.embedding_lookup(self.iid_W, self.pos_items)

        pos_i_e = pos_i_e + k_test
        u_e = tf.nn.embedding_lookup(self.uid_W, self.users)

        dot = tf.einsum('ac,bc->abc', u_e, pos_i_e)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)
        self.batch_predictions = pre

    def _build_graph(self):
        # 创建 占位符
        self._create_placeholders()
        # 创建 可训练的参数
        self._create_variables()
        #
        self._create_inference()
        self._create_loss()
        self._creat_prediction()
