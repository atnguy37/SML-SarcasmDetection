import csv
import json
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer as sb

# Create class
class dataprocess:

    def loadcsv(self, file):
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            data = list(reader)
        return data

    def dataSplit(self, data, comments):
        parents = [x[0].split(' ') for x in data]
        responses = [x[1].split(' ') for x in data]
        labels = [x[2].split(' ') for x in data]
        train_data = {'parent': [], 'response': [], 'labels': labels, 'subreddit': []}
        for idx in range(len(data)):
            train_data['parent'].append([comments[x]['text'] for x in parents[idx]])
            train_data['response'].append([comments[x]['text'] for x in responses[idx]])
            train_data['subreddit'].append([comments[x]['subreddit'] for x in responses[idx]])

        return train_data

    def loadjson(self, file):
        with open(file, 'r') as f:
            comments = json.load(f)
        return comments

    def writejson(self, file, data):
        with open(file, 'w') as f:
            json.dump(data, f)

    def filedump(self, file, filepath):
        data = self.loadcsv(filepath)
        comments = self.loadjson('comments.json')
        dump_data = self.dataSplit(data, comments)
        self.writejson(file, dump_data)

    # Function to Fetch the Glove vectors and load them into memory
    def fetchglove(self):

        glove_wordmap = {}
        # download Glove vectors from directory
        with open("glove.6B.100d.txt", encoding='utf-8') as file:
            for line in file:
                word, vector = tuple(line.split(" ", 1))
                glove_wordmap[word] = np.fromstring(vector, sep=' ')

        # Return the dictionary containing the vector representations of words
        return glove_wordmap

    def cleanup(self, sentence):
        sentence = re.sub("[^A-Za-z0-9^,!.\/'+-=]", " ", sentence)
        sentence = re.sub(r"what's", "what is", sentence)
        sentence = re.sub(r"\'s", " ", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)

        return sentence

    def sentence2sequence(self,sentence, glove_map, limit, vec_len = 100,combine_flag = False):
        sentence = self.cleanup(sentence.lower())
        tokens = sentence.split(' ')
        # Remove Stop words
        stop = set(stopwords.words("english"))
        tokens = [w for w in tokens if not w in stop and len(w) >= 3]
        # Perform Stemming
        tokens = [sb('english').stem(w) for w in tokens]
        # Limit the number of words in a sentence to 50
        if len(tokens) > limit:
            tokens = tokens[:limit]
        elif len(tokens) == 0:
            if combine_flag == False:
                return np.zeros(shape=(limit, vec_len)).astype(np.float32)
            else:
                return np.zeros(shape=(limit * vec_len)).astype(np.float32)
        vecs = []
        # Fetch the Glove vectors for each of the words
        for token in tokens:
            try:
                vec = np.array(glove_map[token]).astype(np.float32)
            except KeyError:
                vec = np.random.uniform(-0.25,0.25, vec_len).astype(np.float32)

            if combine_flag == False:
                vecs.append(vec)
            else:
                # print("OKKKKKKKKKKKKKKK")
                vecs.extend(vec)

        return vecs

    def combine(self,data):
        sentences = []
        # Concatenate the responses and the latest parent
        idx = 0
        for elem in data['parent']:
            for res in data['response'][idx]:
                # print(elem[-1])
                sentences.append(elem[-1] + ' ' + res)
            idx = idx + 1
        # for idx in range(len(data['response'])):
        #     print(idx)
        #     temp1 = data['parent'][idx][-1] + ' ' + data['response'][idx][0]
        #     temp2 = data['parent'][idx][-1] + ' ' + data['response'][idx][1]
        #     sentences.append(temp1)
        #     sentences.append(temp2)
        # Flatten the labels list
        labels = [x for y in data['labels'] for x in y]
        print(np.array(labels).shape)
        print(len(sentences))
        return sentences, labels

    def fit_to_size(self, matrix, shape,combine_flag = False):
        res = np.zeros(shape).astype(np.float32)
        # print(res.shape)
        # print(np.array(matrix).shape)
        if combine_flag == False:
            slices = tuple([slice(0, min(dim, shape[e])) for e, dim in enumerate(matrix.shape)])
        else:
            slices = slice(0, min(len(matrix), shape))
        # print(res.shape)
        # print(matrix)
        res[slices] = matrix[slices]
        return res

    def score_setup(self,row):
        score = np.zeros((2,))
        if row == 0:
            score[0] += 1
        else:
            score[1] += 1

        return score

    def get_vocab_size(self, train):
        train = [x for y in train for x in y]
        train = set(train)
        return len(train)

    def get_user_stats(self, data):
        label = data['labels']
        users = data['subreddit']
        label = [x for y in label for x in y]
        users = [x for y in users for x in y]

        users_set = set(users)
        user_dict = {key: [0,0] for key in users_set}

        for idx in range(len(label)):
            if label[idx] == '0':
                user_dict[users[idx]][1] += 1
            else:
                user_dict[users[idx]][0] += 1

        for user in users_set:
            user_dict[user][1] = user_dict[user][1] / user_dict[user][1] + user_dict[user][0]
            user_dict[user][0] = user_dict[user][0] / user_dict[user][1] + user_dict[user][0]
        print(user_dict)
        return user_dict


    def dump_users(self, user1, user2, filepath):
        print(user1)
        key1 = list(user2.keys())
        key2 = list(user1.keys())
        print(len(key1))
        print(len(key2))
        keys = key1 + key2
        keys = set(keys)
        print(len(keys))
        exit(-1)
        users = {key: 0 for key in keys}

        for key in keys:
            if (key in user1.keys()) and (key in user2.keys()):
                users[key] = (user1[key][0] + user2[key][0]) / (
                            user1[key][0] + user2[key][0] + user1[key][1] + user2[key][1])
            elif key in user1.keys():
                users[key] = (user1[key][0]) / (user1[key][0] + user1[key][1])
            elif key in user2.keys():
                users[key] = (user2[key][0]) / (user2[key][0] + user2[key][1])

        self.writejson(filepath, users)

# dp = dataprocess()
# # dp.filedump('test_dataUB.json', 'test-unbalanced.csv')
# # dp.filedump('train_dataUB.json', 'train-unbalanced.csv')
# # print('Files dumped')
# data1 = dp.loadjson('test_dataUB.json')
# data2 = dp.loadjson('train_dataUB.json')
#
# user1 = dp.get_user_stats(data1)
# user2 = dp.get_user_stats(data2)
# # dp.dump_users(user1, user2, 'users.json')
# print('Users dumped')



# dp.filedump('test_dataUB.json','test-unbalanced.csv')
