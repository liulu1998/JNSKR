import os
import numpy as np
import pandas as pd


def build_train_set(train_set, n_users, relation_set, tail_set, n_relations, n_entities):
    """ train_set, relation_set, tail_set 对齐, 填充值为 n_relations, n_entities

    Returns: 填充过的 train_set, relation_set, tail_set
    Returns: item 与 user 的最大交互数
    Returns: entity 连接的最大 relation 数
  """
    # 每个 item 交互的 user 个数
    user_lenth = []

    # train_set : item -> [user1, user2 ...]
    for i in train_set:
        user_lenth.append(len(train_set[i]))
    user_lenth.sort()
    # 取最后一个元素（最大值）
    # max_user_pi : item 与 user 的最大交互数
    max_user_pi = user_lenth[int(len(user_lenth) * 0.9999)]

    # train_set :dict, item -> [user1, user2 ...]
    # 对齐成长度相同
    for i in train_set:
        # 过长截断
        if len(train_set[i]) > max_user_pi:
            train_set[i] = train_set[i][0: max_user_pi]
        # 不足则补齐, 填充值为 n_users (不存在 ID 为 n_users 的 用户)
        while len(train_set[i]) < max_user_pi:
            train_set[i].append(n_users)

    relation_lenth = []
    for i in relation_set:
        relation_lenth.append(len(relation_set[i]))
    relation_lenth.sort()
    # entity 与 relation 的最大连接数
    max_relation_pi = relation_lenth[int(len(relation_lenth) * 0.9999)]

    # 对齐
    for i in relation_set:
        # 过长截断
        if len(relation_set[i]) > max_relation_pi:
            relation_set[i] = relation_set[i][0:max_relation_pi]
            tail_set[i] = tail_set[i][0:max_relation_pi]
        # 不足则填充，填充值为 n_relations (不存在 ID 为 n_relation 的 关系)
        # 不足则填充, 填充值为 n_entities (不存在 ID 为 n_entities 的 实体)
        while len(relation_set[i]) < max_relation_pi:
            relation_set[i].append(n_relations)
            tail_set[i].append(n_entities)

    # print(max_relation_pi)
    return train_set, relation_set, tail_set, max_user_pi, max_relation_pi


def calculate_weight(c0, c1, p, train_set, train_len, relation_train, relation_len):
    """ TODO calculate_weight ?

    Args:
        c0:
        c1:
        p:
        train_set:
        train_len:
        relation_train:
        relation_len:

    Returns:

    """
    m = [0] * len(train_set.keys())
    for i in train_set.keys():
        m[i] = len(train_set[i]) * 1.0 / train_len

    c = [0] * len(train_set.keys())
    tem = 0
    for i in train_set.keys():
        tem += np.power(m[i], p)
    for i in train_set.keys():
        c[i] = c0 * np.power(m[i], p) / tem

    mk = [0] * len(relation_train.keys())
    for i in relation_train.keys():
        mk[i] = len(relation_train[i]) * 1.0 / relation_len

    ck = [0] * len(relation_train.keys())
    tem = 0
    for i in relation_train.keys():
        tem += np.power(mk[i], p)
    for i in relation_train.keys():
        ck[i] = c1 * np.power(mk[i], p) / tem

    # print(c[0:10])
    # print(ck[0:10])

    c = np.array(c)
    ck = np.array(ck)
    return c, ck


def load_data(DATA_ROOT, args):
    train_file = os.path.join(DATA_ROOT, 'train.txt')
    test_file = os.path.join(DATA_ROOT, 'test.txt')

    # number of users, number of items
    n_users, n_items = 0, 0
    # len of train set
    train_len = 0

    # >>> 统计 n_users, n_items, train_len
    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                # current user id
                uid = int(l[0])
                # 假设 item, user 的 ID 都是连续的
                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
                train_len += len(items)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                n_items = max(n_items, max(items))
    # 因为 user, item ID 从 0 开始, num + 1
    # 这 2 个值也用来填充（对齐）
    n_items += 1
    n_users += 1
    # <<< 统计 n_items, n_users, train_len

    # TODO kg_final2 与 kg_final 的区别
    # h, r, t
    tp_kg = pd.read_csv(os.path.join(DATA_ROOT, 'kg_final2.txt'), sep=' ', header=None)

    # df 的 h 列, r 列, t 列
    head_train = np.array(tp_kg[0], dtype=np.int32)
    relation_train = np.array(tp_kg[1], dtype=np.int32)
    tail_train = np.array(tp_kg[2], dtype=np.int32)

    # item ID -> [user1, user2, ...], 得到 item 和与它交互过的所有 user
    train_set = {}

    with open(train_file) as f_train:
        for l in f_train.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            # user ID, items
            uid, train_items = items[0], items[1:]
            for i in train_items:
                # i -> item
                if i in train_set:
                    train_set[i].append(uid)
                else:
                    train_set[i] = [uid]

    # relation_set: head entity -> [r1, r2 ...] , 即 h 引出的 所有关系 (边)
    # tail_set: head entity -> [t1, t2 ...], 即与 h 邻接的所有 t (节点)
    relation_set, tail_set = {}, {}

    # head_train: KG 中的 head entities
    for i in range(len(head_train)):
        cur_h = head_train[i]
        if cur_h in relation_set:
            # relation_train[i] 是 第 i 个关系, 从 DataFrame 中取的, 和 cur_h 在同一行
            # tail_train[i] 是 第 i 个尾实体, 从 DataFrame 中取的, 和 cur_h 在同一行
            relation_set[cur_h].append(relation_train[i])
            tail_set[cur_h].append(tail_train[i])
        else:
            relation_set[cur_h] = [relation_train[i]]
            tail_set[cur_h] = [tail_train[i]]

    # 假设 relation 连续的
    n_relations = max(relation_train) + 1
    n_entities = max(tail_train) + 1

    # TODO ? relation_len 为什么是 fact 的数量
    relation_len = len(head_train)

    """
    c0 and c1 determine the overall weight of non-observed instances in implicit feedback data.
    Specifically, c0 is for the recommendation task and c1 is for the knowledge embedding task.
    """
    negative_c, negative_ck = calculate_weight(
        args.c0, args.c1, args.p, train_set, train_len, relation_set, relation_len
    )

    train_set, relation_set, tail_set, max_user_pi, max_relation_pi = \
        build_train_set(train_set, n_users, relation_set, tail_set, n_relations, n_entities)

    return n_users, n_items, n_relations, n_entities, train_set, relation_set, tail_set, \
           max_user_pi, max_relation_pi, negative_c, negative_ck


def get_count(tp, id_):
    playcount_groupbyid = tp[[id_]].groupby(id_, as_index=False)
    count = playcount_groupbyid.size()
    return count


def get_train_instances(train_set, relation_set, tail_set):
    """
    Args:
        train_set: dict, item -> [user1, user2 ...]
        tail_set: dict, item(entity) -> [t1, t2 ...]
        relation_set: dict, item(entity) -> [r1, r2 ...]

    Returns:

    """
    item_train, user_train, relation_train1, tail_train1 = [], [], [], []

    for i in relation_set.keys():
        # i = item
        if i in train_set.keys():
            # i 是 一个整数
            item_train.append(i)
            # train_set[i] ... 都是 list
            user_train.append(train_set[i])
            relation_train1.append(relation_set[i])
            tail_train1.append(tail_set[i])
        # <<< if
    # <<< for

    item_train = np.array(item_train)

    user_train = np.array(user_train)
    relation_train1 = np.array(relation_train1)
    tail_train1 = np.array(tail_train1)

    # item_train 增加一个维度, 变为 2D Tensor
    item_train = item_train[:, np.newaxis]
    # 返回的都是 2D Tensor
    return item_train, user_train, relation_train1, tail_train1


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
