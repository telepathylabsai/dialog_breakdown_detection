from breakdown_detection.settings import (
    SENTENCE_ENCODER_MODEL_NAME)

from sentence_transformers import SentenceTransformer


def test_encode():
    sentence_encoder = SentenceTransformer(
        'sentence-transformers/{}'.format(SENTENCE_ENCODER_MODEL_NAME))

    entities = ['car_make', 'car_model', 'car_year']
    ent_sent = sentence_encoder.tokenizer.sep_token.join(
        entities).replace('_', ' ')
    ent_sent
    enc_dict = sentence_encoder.tokenize([ent_sent])

    decoded = sentence_encoder.tokenizer.convert_ids_to_tokens(
        enc_dict['input_ids'][0])
    assert decoded == [
        '[CLS]', 'car', 'make', '[SEP]', 'car', 'model', '[SEP]',
        'car', 'year', '[SEP]']

    enc_dict = sentence_encoder.tokenize([" "])
    decoded = sentence_encoder.tokenizer.convert_ids_to_tokens(
        enc_dict['input_ids'][0])
    assert decoded == ['[CLS]', '[SEP]']
