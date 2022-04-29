import json
import boto3
import string
import sys
import numpy as np
from botocore.exceptions import ClientError

from hashlib import md5

vocabulary_length = 9013

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]


def ses(message, email):
    SENDER = "Name <tw2221@assignment33.awsapps.com>"
    RECIPIENT = email
    
    # CONFIGURATION_SET = "ConfigSet"
    AWS_REGION = "us-east-1"
    SUBJECT = "The spam detector result of your email. "
    BODY_TEXT = (message)
    CHARSET = "UTF-8"
    client = boto3.client('ses',region_name=AWS_REGION)
    
    try:
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong. 
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
        
def readEmail(email):
    lines = email.split('\n')
    Date = lines[0][5:]
    To = lines[1][3:]
    From = lines[2][5:]
    Subject = lines[3][8:]
    Body = '\n'.join(lines[4:])
    result = {}
    result['Date'] = Date
    result['To'] = To
    result['From'] = From
    result['Subject'] = Subject
    result['Body'] = Body
    return result
    
    

s3 = boto3.client('s3')
S3_BUCKET = 's1-ass3'
S3_PREFIX = 'email'

ENDPOINT_NAME ='sms-spam-classifier-mxnet-2022-04-28-18-37-07-454'

runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print(event)
    object_key = event['Records'][0]['s3']['object']['key']
    
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET, Prefix=S3_PREFIX, StartAfter=S3_PREFIX,)
    s3_files = response["Contents"]
    file_content = s3.get_object(
        Bucket=S3_BUCKET, Key=object_key)["Body"].read().decode()
    print(file_content)
    info = readEmail(file_content)
    print(info)
    file_content = [file_content.strip()]

    test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    one_hot_test_messages = one_hot_encode(file_content, 9013)
    print('oht done')
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, 9013)
    print('etm done')

    data = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=data)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = int(result['predicted_label'][0][0])
    predicted_label = 'delay' if pred == 1 else 'no delay'
    info = buildEmail(info, predicted_label, pred)
    print(info)
    ses(info, 'nisetamago233@gmail.com')
        
    return predicted_label
    
    
def buildEmail(info, classification, score):
    result = 'We received your email sent at {Date} with the subject {Subject} Here is a 240 character sample of the email body: \n{Body} The email was categorized as {Classification} with a {Score:.2%} confidence.'
    result = result.format(Date=info['Date'],
                           Subject = info['Subject'],
                           Body = info['Body'],
                           Classification = classification,
                           Score = score)
    return result