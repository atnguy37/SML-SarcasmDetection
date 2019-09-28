import DataPreprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell
from tqdm import tqdm
from attention import attention_mechanism
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

class ATT(DataPreprocess.dataprocess):
    def __init__(self, batch_size=256, lstm_size=64, vector_size=100, max_seq=30, learn_rate=0.01, output_p=0.5, num_layers = 2, epoch = 10):
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.vector_size = vector_size
        self.max_seq = max_seq
        self.learn_rate = learn_rate
        self.output_p = output_p
        self.num_layers = num_layers
        self.epoch = epoch

    # Method to fetch the data
    def fetchdata(self, file, filepath='', dump_flag = False, combine_flag = False, w2vflg = False):
        if dump_flag is True:
            self.filedump(file, filepath)

        data = self.loadjson(file)

        # Check if the responses and parent have to combined or not
        if combine_flag is True:
            sentences, labels = self.combine(data)
        else:
            context = []
            idx = 0
            for elem in data['parent']:
                for _ in range(len(data['response'][idx])):
                    context.append(elem[-1])
                idx = idx + 1
            response = [x for y in data['response'] for x in y]
            labels = [x for y in data['labels'] for x in y]


        # ##### Testing purpose ####
        # response = response[0:128]
        # context = context[0:128]
        # labels = labels[0:128]

        labels = list(map(int, labels))
        labels = np.array(list(map(self.score_setup, labels)))

        return context, response, labels

        # if w2vflg is False:
        #     # Fetch the Glove vectors
        #     glove_map = self.fetchglove()
        # else:
        #     # Fetch the word2vec
        #     glove_map = Word2Vec.load('w2vmodel')
        #
        # # Convert the sentences into a sequence
        # response = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in response]
        # response = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in response])
        #
        # context = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in context]
        # context = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in context])
        #
        # print('Word Embeddings have been fetched')
        #
        # if combine_flag is True:
        #     return sentences, labels
        # else:
        #     return context, response, labels

    def convert(self, context, response, glove_map):
        # Convert the sentences into a sequence
        response = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in response]
        response = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in response])

        context = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in context]
        context = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in context])

        return context, response



    # Method to fit the ATM Model
    def fit(self, context, response, labels,context_test = None, response_test = None , labels_test = None, att_flag = True, test_flg = False, w2vflg=True):
        # Create Placeholders
        label_ph = tf.placeholder(tf.float32, [None, 2])
        resp_ph = tf.placeholder(tf.float32, [None, self.max_seq, self.vector_size])
        cont_ph = tf.placeholder(tf.float32, [None, self.max_seq, self.vector_size])

        weight = tf.Variable(tf.random_normal([2 * self.lstm_size, 2]))
        bias = tf.Variable(tf.constant(0.1, shape=[2]))

        
        if att_flag is False:
            mlstmCell_fw = []
            mlstmCell_bw = []

            # RNN output weights and bias
            input_ph = tf.concat([cont_ph, resp_ph], 1)
            input_ph = tf.transpose(input_ph, [1, 0, 2])
            input_ph = tf.reshape(input_ph, [-1, self.vector_size])

            input_ph = tf.split(input_ph, self.max_seq * 2)

            # Create the LSTM Cell
            for _ in range(self.num_layers):
                lstmCell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
                lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.output_p)
                mlstmCell_fw.append(lstmCell)
                mlstmCell_bw.append(lstmCell)

            mlstmCell_fw = tf.contrib.rnn.MultiRNNCell(cells=mlstmCell_fw)
            mlstmCell_bw = tf.contrib.rnn.MultiRNNCell(cells=mlstmCell_bw)

            rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mlstmCell_fw, mlstmCell_bw, input_ph,
                                                                        dtype=tf.float32)
            #last = tf.gather(value, int(rnn_outputs.get_shape()[0]) - 1)
            last = rnn_outputs[-1]
        # The Attention Based Bi Directional LSTM Method
        else:

            input_ph = tf.concat([cont_ph, resp_ph], 1)
            lstm_output, _ = bidirectional_dynamic_rnn(GRUCell(self.lstm_size), GRUCell(self.lstm_size), inputs=input_ph,
                                                     dtype=tf.float32)
            outputs, alphas = attention_mechanism.attention(lstm_output, 50, return_alphas=True)
            outputs = tf.nn.dropout(outputs, self.output_p)
            last = outputs

        prediction = (tf.matmul(last, weight) + bias)
        # Prediction
        correctPred = tf.equal(tf.cast(tf.argmax(prediction, 1), 'int32'), tf.cast(tf.argmax(label_ph, 1), 'int32'))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=label_ph))

        optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        # Function to save the current trained model
        saver = tf.train.Saver()
        sess = tf.Session()
        training_iterations = int(len(response) / self.batch_size)
        # test_iterations = int(len(response_test) / self.batch_size)
        # training_iterations = range(0, len(response), self.batch_size)
        # training_iterations = tqdm(training_iterations)
        test_accuracy = []
        test_confuse = [[0,0],[0,0]]

        ## For Plotting
        test_accuracy = []
        accuracy_plot = []
        train_error_plot = []
        loss_error_plot = []
        train_plot = []
        loss_plot = []
        x_test = []
        x_train = []

        train_plot_epoch = []
        loss_plot_epoch = []
        x_test = []
        x_train_epoch = []

        if w2vflg is False:
            # Fetch the Glove vectors
            glove_map = self.fetchglove()
        else:
            # Fetch the word2vec
            glove_map = Word2Vec.load('w2vmodel')

        if test_flg is False:
            # Training
            init = tf.global_variables_initializer()
            sess.run(init)
            # batch = 0
            # for e in range(self.epoch):
            for i in range(training_iterations):
                #batch = np.random.randint(len(response), size=self.batch_size)
                # if i != 0:
                #     prev = batch
                #     batch = prev + self.batch_size
                # else:
                #     prev = 0
                #     batch = self.batch_size

                cont, resp = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], response[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, resp_p, label_p = (cont, resp, labels[i*self.batch_size:(i+1)*self.batch_size])

                sess.run([optimizer], feed_dict={cont_ph: cont_p, resp_ph: resp_p, label_ph: label_p})
                if i % 10 == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={cont_ph: cont_p, resp_ph: resp_p, label_ph: label_p})
                    # Calculate batch loss
                    tmp_loss = sess.run(loss, feed_dict={cont_ph: cont_p, resp_ph: resp_p, label_ph: label_p})
                    # Display results
                    print("Iter " + str(i) + ", Minibatch Loss= " + \
                        "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))
                    train_plot.append(acc)
                    loss_plot.append(tmp_loss)
                    x_train.append(i)
                    # if(i < test_iterations):
                    cont, resp = self.convert(context_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size], response_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size], glove_map)
                    cont_p, resp_p, label_p = (cont, resp, labels_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size])

                    # Calculate batch accuracy
                    acc_test = sess.run(accuracy, feed_dict={cont_ph: cont_p, resp_ph: resp_p, label_ph: label_p})
                    # Calculate batch loss
                    loss_test = sess.run(loss, feed_dict={cont_ph: cont_p, resp_ph: resp_p, label_ph: label_p})
                    print("Iter " + str(i) + ", Minibatch Loss Test= " + \
                    "{:.6f}".format(loss_test) + ", Training Accuracy Test= " + \
                    "{:.5f}".format(acc_test))

                    train_error_plot.append([acc,acc_test])
                    loss_error_plot.append([tmp_loss,loss_test])
                    x_test.append(i)
                        # print(train_error_plot)
                        # print(loss_error_plot)
            
            saver.save(sess, 'ModelEpoch/Final_model_LSTM_AT')
            print('Model saved to Root Directory of the python file')
            np.save("Graph/train_error_plot_LSTM_AT",train_error_plot)
            np.save("Graph/loss_error_plot_LSTM_AT",loss_error_plot)
            np.save("Graph/x_test_LSTM_AT",x_test)
            # np.save("Graph/train_plot",train_plot)
            # np.save("Graph/loss_plot",loss_plot)
            # np.save("Graph/x_train",x_train)
            # self.plot_graph(x_train_epoch, train_plot_epoch, 'LSTM Attention Train Accuracy Plot Epoch', 'Epoch', 'Accuracy')
            # self.plot_graph(x_train_epoch, loss_plot_epoch, 'LSTM Attention Train Loss Plot Epoch', 'Epoch', 'Accuracy')
        else:
            confusion_matrix_tf = tf.confusion_matrix(tf.cast(tf.argmax(label_ph, 1), 'int32'),tf.cast(tf.argmax(prediction, 1), 'int32'))
            saver.restore(sess, 'ModelEpoch/Final_model_LSTM')
            # batch = 0
            print(training_iterations)

            for i in range(training_iterations):
                # if i != 0:
                #     prev = batch
                #     batch = prev + self.batch_size
                # else:
                #     prev = 0
                #     batch = self.batch_size
                cont, resp = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], response[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, resp_p, label_p = (cont, resp, labels[i*self.batch_size:(i+1)*self.batch_size])
                predict = sess.run(accuracy, feed_dict={cont_ph: cont_p,
                                                          resp_ph: resp_p,
                                                          label_ph: label_p})
                cm = sess.run(confusion_matrix_tf, feed_dict={cont_ph: cont_p,
                                                          resp_ph: resp_p,
                                                          label_ph: label_p})
                if i % 10 == 0:
                    print("Iter " + str(i) +  ", Training Accuracy= " + \
                          "{:.5f}".format(predict))
                # if (i/self.batch_size) % 10 == 0:
                #     accuracy_plot.append(predict)
                #     x_test.append(i)
                test_confuse = np.add(test_confuse, cm)
                test_accuracy.append(predict)

            ## For the Plot
            # self.plot_graph(x_test, accuracy_plot, 'LSTM Test Accuracy Plot')
            print("Accuracy: ",float(sum(test_accuracy)/len(test_accuracy)))
            precision = test_confuse[1][1] / (test_confuse[0][1] + test_confuse[1][1])
            recall = test_confuse[1][1] / (test_confuse[1][0] + test_confuse[1][1])
            print("Confuse: ")
            print(test_confuse)
            print("Precision: ",precision)
            print("Recall: ",recall)
            print("F1-Score: ",2*(precision * recall) / (precision + recall))

        sess.close()

    def plot_graph(self, x, y, title,xlabel,ylabel):
        plt.plot(x, y, alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(title + '.png')
        # plt.show()

att = ATT()
train = input("Would you like to train the data set? (Y/N)")
if train == 'Y':
    context, response, labels = att.fetchdata('train_data.json', w2vflg=True)
    context_test, response_test, labels_test = att.fetchdata('test_data.json', w2vflg=True)
    att.fit(context, response, labels, context_test, response_test, labels_test, att_flag = True, w2vflg=True)
elif train == 'N':
    context, response, labels = att.fetchdata('test_data.json', w2vflg=True)
    att.fit(context, response, labels, test_flg=True,  att_flag = False, w2vflg=True)
    #att.test(context, response, labels, prediction)
