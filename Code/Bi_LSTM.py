import DataPreprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell
from tqdm import tqdm
from attention import attention_mechanism
from gensim.models import Word2Vec

class BiLSTM(DataPreprocess.dataprocess):
    def __init__(self, batch_size=256, , vector_size=100, max_seq=30, learn_rate=0.01, output_p=0.5, num_layers = 2, num_hidden = 64):
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.max_seq = max_seq
        self.learn_rate = learn_rate
        self.output_p = output_p
        self.num_layers = num_layers
        self.num_hidden = num_hidden

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



    def convert(self, context, response, glove_map):
        # Convert the sentences into a sequence
        response = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in response]
        response = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in response])

        context = [np.vstack(self.sentence2sequence(sent, glove_map, self.max_seq)) for sent in context]
        context = np.stack([self.fit_to_size(x,(self.max_seq,self.vector_size)) for x in context])

        return context, response

    def bidirectional_rnn_model(self, data, weight, bias):
        splitted_data = tf.unstack(data, axis=1)

        lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_cell1, lstm_cell2, splitted_data, dtype=tf.float32)
        
        # output = outputs[-1]
        
        # w_softmax = tf.Variable(tf.random_normal([self.num_hidden*2, num_labels]))
        # b_softmax = tf.Variable(tf.random_normal([num_labels]))
        logit = tf.matmul(outputs[-1], weight) + bias
        return logit

    # Method to fit the BiLSTM Model
    def fit(self, context, response, labels, context_test = None, response_test = None, labels_test = None, test_flg = False, w2vflg=True):
        # Create Placeholders
        label_ph = tf.placeholder(tf.float32, [self.batch_size,2])
        resp_ph = tf.placeholder(tf.float32, [self.batch_size, self.max_seq, self.vector_size])
        cont_ph = tf.placeholder(tf.float32, [self.batch_size, self.max_seq, self.vector_size])

        weight = tf.Variable(tf.random_normal([self.num_hidden*2, 2]))
        bias = tf.Variable(tf.constant(0.1, shape=[2]))

        input_ph = tf.concat([cont_ph, resp_ph], 1)
        # input_ph = tf.transpose(input_ph, [1, 0, 2])
        # input_ph = tf.reshape(input_ph, [-1, self.vector_size])

        # input_ph = tf.split(input_ph, self.max_seq * 2)

        logits = self.bidirectional_rnn_model(input_ph, weight, bias)
    
        # softmax_op = tf.nn.softmax(logits)
        # prediction = tf.argmax(softmax_op,axis = 1)

        # For Testing scenario
        # if test_flg == True:
        #     return prediction

        # Prediction
        correctPred = tf.equal(tf.cast(tf.argmax(logits, 1), 'int32'), tf.cast(tf.argmax(label_ph, 1), 'int32'))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_ph))

        optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        # Function to save the current trained model
        saver = tf.train.Saver()
        sess = tf.Session()
        training_iterations = int(len(response) / self.batch_size)
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
            file = open("result_BI_UB_W2C.txt","w") 
            print(training_iterations)
            init = tf.global_variables_initializer()
            sess.run(init)
            batch = 0
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
                    # file.write("Iter " + str(i) + ", Minibatch Loss= " + \
                    #       "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                    #       "{:.5f}".format(acc) + "\n" )

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


            np.save("Graph/train_error_plot_Bi_LSTM_W2C",train_error_plot)
            np.save("Graph/loss_error_plot_Bi_LSTM_W2C",loss_error_plot)
            np.save("Graph/x_test_Bi_LSTM_W2C",x_test)

            saver.save(sess, 'ModelEpoch/Final_model_BI_W2C')
            print('Model saved to Root Directory of the python file')
        else:
            confusion_matrix_tf = tf.confusion_matrix(tf.cast(tf.argmax(label_ph, 1), 'int32'),tf.cast(tf.argmax(logits, 1), 'int32'))
            saver.restore(sess, 'ModelEpoch/Final_model_BI')
            batch = 0
            # file = open("result_BI_UB_W2C_test.txt","w") 
            print(training_iterations)
            # file.write(str(training_iterations) + "\n")
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
                    # file.write("Iter " + str(i) +  ", Training Accuracy= " + \
                    #       "{:.5f}".format(predict) + "\n")
                test_confuse = np.add(test_confuse, cm)
                test_accuracy.append(predict)

            print("Accuracy: ",float(sum(test_accuracy)/len(test_accuracy)))
            precision = test_confuse[1][1] / (test_confuse[0][1] + test_confuse[1][1])
            recall = test_confuse[1][1] / (test_confuse[1][0] + test_confuse[1][1])
            print("Confuse: ")
            print(test_confuse)
            print("Precision: ",precision)
            print("Recall: ",recall)
            print("F1-Score: ",2*(precision * recall) / (precision + recall))
            # file.write("Accuracy: " + str(float(sum(test_accuracy)/len(test_accuracy))) + "\n")
            # file.write("Confuse: " + str(test_confuse) + "\n")
            # file.write("Precision: " + str(precision) + "\n")
            # file.write("Recall: " + str(recall) + "\n")
            # file.write("Recall: " + str(2*(precision * recall) / (precision + recall)))
            # file.close()

        sess.close()


biLstm = BiLSTM()
train = input("Would you like to train the data set? (Y/N)")
if train == 'Y':
    context, response, labels = biLstm.fetchdata('train_data.json', w2vflg=False)
    context_test, response_test, labels_test = biLstm.fetchdata('test_data.json', w2vflg=False)
    biLstm.fit(context, response, labels, context_test, response_test, labels_test, w2vflg=False)
elif train == 'N':
    context, response, labels = biLstm.fetchdata('test_data.json', w2vflg=True)
    biLstm.fit(context, response, labels, test_flg=True,  w2vflg=True)
    #biLstm.test(context, response, labels, prediction)