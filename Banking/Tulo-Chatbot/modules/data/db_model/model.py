from mongoengine import *
from modules.utils.yaml_parser import Config
import json
from datetime import datetime

url = Config.get_config_val(key="mongodb", key_1depth="url")
db = Config.get_config_val(key="mongodb", key_1depth="db")
connect(db, host=url)


class User(Document):
    first_name = StringField(required=True, max_length=100)
    last_name = StringField(required=True, max_length=100)
    email = StringField(required=True, max_length=255)
    password = StringField(required=True, max_length=100)
    age = IntField(required=False)
    gender = StringField(required=False, max_length=1)
    created_on = DateTimeField(required=True)
    telegram_oAuth_token = StringField(required=False, max_length=100)


# TODO to use it later. Not yet used. Also extend it from Document
class Language:
    lang_code = StringField(required=True, max_length=8)


class Broker(Document):
    user_id = ReferenceField(User)
    broker_name = StringField(required=True, max_length=100)
    default_lang = StringField(required=True, max_length=8, default="en-US")    # TODO replace with Language


# Relationship is as below
# User (1) : Broker (n)
# Broker (1) : Language (n) : TrainedClassifier(1)
# So every user can create multiple brokers, each broker may have multiple classifiers, (1 per language)
class Trainedclassifier(Document):
    user = ReferenceField(User)  # denormalized for easier usage. Otherwise not required
    broker = ReferenceField(Broker)
    model_type = StringField(max_length=100)
    vector_type = StringField(max_length=100)
    classifier = BinaryField()
    lang = StringField(required=True, max_length=8)    # TODO replace with Language


class Circumstance(EmbeddedDocument):
    input_circumstance = StringField(required=False, max_length=100)
    output_circumstance = StringField(required=False, max_length=100)


class Variables(EmbeddedDocument):
    name = StringField(required=True, max_length=100)  # number_1
    type = StringField(required=True, max_length=100)  # @def.number
    value = StringField(required=True, max_length=100)  # $number1
    io_type = StringField(required=True, max_length=100)  # @def.input


class Response(EmbeddedDocument):
    text = ListField(StringField(required=False, max_length=1000))
    custom = StringField(required=False)


# This is the final class
class Train(Document):
    trained_classifier = ReferenceField(Trainedclassifier)
    lang = StringField(required=True)    # TODO replace with Language
    category = StringField(required=True)
    circumstance = EmbeddedDocumentField('Circumstance')
    variables = ListField(EmbeddedDocumentField('Variables'))
    training_queries = ListField(StringField(required=True, max_length=1000))
    response = ListField(EmbeddedDocumentField('Response'))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Unclassifiedquery(Document):
    trained_classifier = ReferenceField(Trainedclassifier, required=True)       # This is the classifier against which this query was raised
    created_on = DateTimeField(required=True, default=datetime.now())           # created on which date
    query = StringField(required=True, max_length=300)                          # what was the query
    is_trained = BooleanField(required=True, default=False)                     # whether the user has trained it or not
    trained_on = DateTimeField(required=False)                                  # which date was it trained on
    train_category = ReferenceField(Train, required=False)                      # which category does it belong to after training
    is_discardable = BooleanField(required=True, default=False)                 # whether the data is junk or can be trained
