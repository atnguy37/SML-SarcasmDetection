import DataPreprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell
from tqdm import tqdm
from attention import attention_mechanism
from gensim.models import Word2Vec

class SVM(DataPreprocess.dataprocess):
    def __init__(self, batch_size=256, vector_size=100, max_seq=25, learn_rate=0.1, output_p=0.5, num_layers = 1, num_hidden = 64):
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
            labels_SVM = []
            for x in labels:
                if(int(x) == 0):
                    labels_SVM.append(-1)
                else:
                    labels_SVM.append(1)
            return np.array(sentences), np.array(labels_SVM)
        else:
            context = []
            idx = 0
            for elem in data['parent']:
                for _ in range(len(data['response'][idx])):
                    context.append(elem[-1])
                idx = idx + 1
            response = [x for y in data['response'] for x in y]
            labels = [x for y in data['labels'] for x in y]


        return context, response, labels


    def convert(self, context, glove_map):
        # Convert the sentences into a sequence

        context = np.stack([self.fit_to_size(self.sentence2sequence(sent, glove_map, 2* self.max_seq,combine_flag = True),
            self.max_seq * self.vector_size * 2,combine_flag = True) for sent in context])

        return context

    # Method to fit the SVM Model
    def fit(self, context, labels, context_test = None, labels_test = None, test_flg = False, w2vflg=True):
        # Create Placeholders
        x_data = tf.placeholder(shape=[None, self.max_seq * self.vector_size * 2], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, self.num_layers], dtype=tf.float32)
        A = tf.Variable(tf.random_normal(shape=[self.max_seq * self.vector_size * 2,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))

        model_output = tf.subtract(tf.matmul(x_data, A), b)
        l2_norm = tf.reduce_sum(tf.square(A))
        alpha = tf.constant([0.1])
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1.,tf.multiply(model_output, y_target))))
        loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target),tf.float32))
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_ph))

        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)
        new_prediction = tf.reshape(prediction,[-1])
        shape = tf.shape(new_prediction)
        # new_prediction = [0 if x==-1 else 1 for x in new_prediction]
        new_prediction = tf.map_fn(lambda x: tf.cond(x < 0, lambda: x+1, lambda: x), new_prediction)
        new_prediction = tf.cast(tf.reshape(new_prediction, shape),'int32')
        new_y = tf.reshape(y_target,[-1])
        # new_y = [0 if x==-1 else 1 for x in new_y]
        new_y = tf.map_fn(lambda x: tf.cond(x < 0, lambda: x+1, lambda: x), new_y)
        new_y = tf.cast(tf.reshape(new_y, shape),'int32')

        # Function to save the current trained model
        saver = tf.train.Saver()
        sess = tf.Session()
        training_iterations = int(len(context) / self.batch_size)
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
            batch = 0
            for i in range(training_iterations):
                #batch = np.random.randint(len(response), size=self.batch_size)
                # if i != 0:
                #     prev = batch
                #     batch = prev + self.batch_size
                # else:
                #     prev = 0
                #     batch change value in tf array= self.batch_size

                cont = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, label_p = (cont, np.transpose([labels[i*self.batch_size:(i+1)*self.batch_size]]))

                sess.run([optimizer], feed_dict={x_data: cont_p, y_target: label_p})
                if i % 10 == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x_data: cont_p, y_target: label_p})
                    # Calculate batch loss
                    tmp_loss = sess.run(loss, feed_dict={x_data: cont_p, y_target: label_p})
                    # Display results
                    # print(tmp_loss)
                    print("Iter " + str(i) + ", Minibatch Loss= " + \
                          "{:.6f}".format(tmp_loss[0]) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))


                    train_plot.append(acc)
                    loss_plot.append(tmp_loss)
                    x_train.append(i)
                    # if(i < test_iterations):
                    cont = self.convert(context_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size], glove_map)
                    cont_p, label_p = (cont, np.transpose([labels_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size]]))

                    # Calculate batch accuracy
                    acc_test = sess.run(accuracy, feed_dict={x_data: cont_p,  y_target: label_p})
                    # Calculate batch loss
                    loss_test = sess.run(loss, feed_dict={x_data: cont_p, y_target: label_p})
                    print("Iter " + str(i) + ", Minibatch Loss Test= " + \
                    "{:.6f}".format(loss_test[0]) + ", Training Accuracy Test= " + \
                    "{:.5f}".format(acc_test))

                    train_error_plot.append([acc,acc_test])
                    loss_error_plot.append([tmp_loss[0],loss_test[0]])
                    x_test.append(i)


            np.save("Graph/train_error_plot_SVM_W2C",train_error_plot)
            np.save("Graph/loss_error_plot_SVM_W2C",loss_error_plot)
            np.save("Graph/x_test_SVM_W2C",x_test)
            saver.save(sess, 'ModelEpoch/Final_model_SVM_W2C')
            print('Model saved to Root Directory of the python file')
        else:
            confusion_matrix_tf = tf.confusion_matrix(new_y, new_prediction)
            saver.restore(sess, 'ModelEpoch/Final_model_SVM_W2C')
            batch = 0
            print(training_iterations)

            for i in range(training_iterations):
                # if i != 0:
                #     prev = batch
                #     batch = prev + self.batch_size
                # else:
                #     prev = 0
                #     batch = self.batch_size
                cont = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, label_p = (cont, np.transpose([labels[i*self.batch_size:(i+1)*self.batch_size]]))
                predict = sess.run(accuracy, feed_dict={x_data: cont_p, y_target: label_p})
                cm = sess.run(confusion_matrix_tf, feed_dict={x_data: cont_p, y_target: label_p})
                if i % 10 == 0:
                    print("Iter " + str(i) +  ", Training Accuracy= " + \
                          "{:.5f}".format(predict))
                    # print(cm)
                test_confuse = np.add(test_confuse, cm)
                test_accuracy.append(predict)
                    # print(predict)

            print("Accuracy: ",float(sum(test_accuracy)/len(test_accuracy)))
            precision = test_confuse[1][1] / (test_confuse[0][1] + test_confuse[1][1])
            recall = test_confuse[1][1] / (test_confuse[1][0] + test_confuse[1][1])
            print("Confuse: ")
            print(test_confuse)
            print("Precision: ",precision)
            print("Recall: ",recall)
            print("F1-Score: ",2*(precision * recall) / (precision + recall))

        sess.close()


svm = SVM()
train = input("Would you like to train the data set? (Y/N)")
if train == 'Y':
    context, labels = svm.fetchdata('train_data.json', combine_flag = True, w2vflg=False)
    context_test, labels_test = svm.fetchdata('test_data.json', combine_flag = True, w2vflg=False)
    # context = np.array(context)
    # labels = np.array(labels)
    # print(context.shape)
    # print(labels.shape)
    # context_test, response_test, labels_test = svm.fetchdata('test_data.json')
    svm.fit(context, labels, context_test, labels_test, test_flg = False, w2vflg=False)
elif train == 'N':
    context, labels = svm.fetchdata('test_data.json', w2vflg=False, combine_flag = True)
    svm.fit(context, labels, test_flg=True, w2vflg=False)
    #svm.test(context, response, labels, prediction)
