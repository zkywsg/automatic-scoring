# -*- coding: utf-8 -*-


import argparse
import time

from create_model import build_model
from evaluator import Evaluator
import asap_reader as dataset
from keras.preprocessing import sequence

from topic_vector import *

logger = get_logger("combine two model")
np.random.seed(100)

is_training = True

def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default='glove.6B.50d.txt', help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")


    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", default='random', required=False)
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value', default=0.001)
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint1 directory', default='./checkpoint')

    parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='attsum',
                        help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234,
                        help="Random seed (default=1234)")
    parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=100,
                        help="RNN dimension. '0' means no RNN layer (default=300)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                        help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>',
                        help="(Optional) The path to the existing vocab file (*.pkl)")

    fold = '4'


    parser.add_argument('--train', default='./data/fold_' + fold + '/train.tsv')  # data_path"
    parser.add_argument('--dev', default='./data/fold_' + fold + '/dev.tsv')
    parser.add_argument('--test', default='./data/fold_' + fold + '/test.tsv')
    parser.add_argument('--prompt_id', type=int, default=3, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true',
                        help='init the last layer bias with average score of training data')

    print('pp3, f' + fold)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_path

    embedding_path = args.embedding_dict
    embedding = args.embedding
    embedd_dim = args.embedding_dim



    (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (
    test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
        (args.train, args.dev, args.test), args.prompt_id, args.vocab_size, args.maxlen,
        tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None)

    embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
    embedd_matrix = build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)

    train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)

    import keras.backend as K

    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())

    train_pmt = np.array(train_pmt, dtype='int32')
    dev_pmt = np.array(dev_pmt, dtype='int32')
    test_pmt = np.array(test_pmt, dtype='int32')

    train_mean = train_y.mean(axis=0)
    scaled_train_mean = dataset.get_model_friendly_scores(train_mean, args.prompt_id)

    train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
    dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
    test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
    logger.info('  train_x shape: ' + str(np.array(train_x).shape))
    logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
    logger.info('  test_x shape:  ' + str(np.array(test_x).shape))

    logger.info('  train_y shape: ' + str(train_y.shape))
    logger.info('  dev_y shape:   ' + str(dev_y.shape))
    logger.info('  test_y shape:  ' + str(test_y.shape))



    model = build_model(args, overal_maxlen, vocab_size,  embedd_dim, embedd_matrix, True, scaled_train_mean)


    evl = Evaluator(args.prompt_id, checkpoint_dir, train_x, dev_x, test_x, train_y, dev_y, test_y)

    # Initial evaluation
    if is_training:
        # logger.info("Initial evaluation: ")
        # evl.evaluate(model, -1, print_info=True)
        logger.info("Train model")
        for ii in xrange(args.num_epochs):

            logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
            start_time = time.time()

            model.fit({'word_input': train_x}, train_y, batch_size=args.batch_size, nb_epoch=1, verbose=0, shuffle=True)
            tt_time = time.time() - start_time
            logger.info("Training one epoch in %.3f s" % tt_time)
            evl.evaluate(model, ii+1)
            evl.print_info()

        evl.print_final_info()

    print('p' + str(args.prompt_id) + ',f' + fold)


if __name__ == '__main__':
    main()

