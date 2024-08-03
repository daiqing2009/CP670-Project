# #   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# #
# #   Licensed under the Apache License, Version 2.0 (the "License");
# #   you may not use this file except in compliance with the License.
# #   You may obtain a copy of the License at
# #
# #         http://www.apache.org/licenses/LICENSE-2.0
# #
# #   Unless required by applicable law or agreed to in writing, software
# #   distributed under the License is distributed on an "AS IS" BASIS,
# #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #   See the License for the specific language governing permissions and
# #   limitations under the License.
# """Prepare TF.Examples for on-device recommendation model.

# Following functions are included: 1) downloading raw data 2) processing to user
# activity sequence and splitting to train/test data 3) convert to TF.Examples
# and write in output location.

# More information about the movielens dataset can be found here:
# https://grouplens.org/datasets/movielens/
# """

import collections
import json
import os
import random
import re
from absl import app, flags, logging
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import chardet

FLAGS = flags.FLAGS

# Permalinks to download movielens data.
RATINGS_FILE_NAME = "ratings.dat"
MOVIES_FILE_NAME = "{content_name}.dat"
RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
OUTPUT_TRAINING_DATA_FILENAME = "train_{content_name}.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_{content_name}.tfrecord"
OUTPUT_MOVIE_VOCAB_FILENAME = "{content_name}_vocab.json"
OUTPUT_MOVIE_YEAR_VOCAB_FILENAME = "{content_name}_year_vocab.txt"
OUTPUT_MOVIE_GENRE_VOCAB_FILENAME = "{content_name}_genre_vocab.txt"
OUTPUT_GENRE_VOCAB_FILENAME = "{content_name}_genre_vocab.json"
OUTPUT_MOVIE_TITLE_UNIGRAM_VOCAB_FILENAME = "music_title_unigram_vocab.txt"
OUTPUT_MOVIE_TITLE_BIGRAM_VOCAB_FILENAME = "music_title_bigram_vocab.txt"
PAD_MOVIE_ID = 0
PAD_RATING = 0.0
PAD_MOVIE_YEAR = 0
UNKNOWN_STR = "UNK"
VOCAB_MOVIE_ID_INDEX = 0
VOCAB_COUNT_INDEX = 3
MAXIUM_GENRES_NUM = 20

def define_flags():
    """Define flags."""
    flags.DEFINE_string("content_name", "movies", "the type of content")
    flags.DEFINE_string("extracted_data_dir", "/tmp", "Path of extracted music data.")
    flags.DEFINE_string("output_dir", None, "Path to the directory of output files.")
    flags.DEFINE_bool("build_vocabs", True, "If yes, generate movie feature vocabs.")
    flags.DEFINE_integer("min_timeline_length", 3, "The minimum timeline length to construct examples.")
    flags.DEFINE_integer("max_context_length", 10, "The maximum length of user context history.")
    flags.DEFINE_integer("max_context_movie_genre_length", 10, "The maximum length of user context history.")
    flags.DEFINE_integer("min_rating", None, "Minimum rating of movie that will be used to in training data")
    flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")
    flags.DEFINE_string("SEPERATOR", "::", "the type of content")

class MovieInfo(collections.namedtuple("MovieInfo", ["movie_id", "timestamp", "rating", "title", "genres"])):
    """Data holder of basic information of a movie."""
    __slots__ = ()

    def __new__(cls, movie_id=PAD_MOVIE_ID, timestamp=0, rating=PAD_RATING, title="", genres=""):
        return super(MovieInfo, cls).__new__(cls, movie_id, timestamp, rating, title, genres)

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def read_data(data_directory, min_rating=None):
    """Read movielens ratings.dat and movies.dat file into dataframe."""
    ratings_file = os.path.join(data_directory, RATINGS_FILE_NAME)
    movies_file = os.path.join(data_directory, MOVIES_FILE_NAME.format(content_name=FLAGS.content_name))
    
    # Detect encoding
    ratings_encoding = detect_encoding(ratings_file)
    movies_encoding = detect_encoding(movies_file)
    
    logging.info(f"Detected encoding for ratings file: {ratings_encoding}")
    logging.info(f"Detected encoding for movies file: {movies_encoding}")
    
    # Read ratings data
    ratings_df = pd.read_csv(
        ratings_file,
        sep=FLAGS.SEPERATOR,
        names=RATINGS_DATA_COLUMNS,
        encoding=ratings_encoding
    )
    ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
    if min_rating is not None:
        ratings_df = ratings_df[ratings_df["Rating"] >= min_rating]
    
    # Read movies data
    movies_df = pd.read_csv(
        movies_file,
        sep=FLAGS.SEPERATOR,
        names=MOVIES_DATA_COLUMNS,
        encoding=movies_encoding
    )
    return ratings_df, movies_df

def convert_to_timelines(ratings_df):
    """Convert ratings data to user timelines."""
    logging.info("Converting ratings to user timelines...")
    timelines = collections.defaultdict(list)
    movie_counts = collections.Counter()
    movie_sum_of_rating = collections.Counter()

    for user_id, movie_id, rating, timestamp in tqdm(ratings_df.values, desc="Processing ratings"):
        timelines[user_id].append(MovieInfo(movie_id=movie_id, timestamp=int(timestamp), rating=rating))
        movie_counts[movie_id] += 1
        movie_sum_of_rating[movie_id] += rating

    logging.info("Sorting user timelines...")
    for user_id, context in tqdm(timelines.items(), desc="Sorting timelines"):
        context.sort(key=lambda x: x.timestamp)
        timelines[user_id] = context

    return timelines, movie_counts, movie_sum_of_rating

def generate_movies_dict(movies_df):
    """Generates movies dictionary from movies dataframe."""
    movies_dict = {
        movie_id: MovieInfo(movie_id=movie_id, title=title, genres=genres)
        for movie_id, title, genres in movies_df.values
    }
    movies_dict[0] = MovieInfo()
    return movies_dict

def extract_year_from_title(title):
    year = re.search(r"\((\d{4})\)", title)
    if year:
        return int(year.group(1))
    return 0

def generate_feature_of_movie_years(movies_dict, movies):
    """Extracts year feature for movies from movie title."""
    return [extract_year_from_title(movies_dict[movie.movie_id].title) for movie in movies]

def generate_movie_genres(movies_dict, movies):
  """Create a feature of the genre of each movie.

  Save genre as a feature for the movies.

  Args:
    movies_dict: Dict of movies, keyed by movie_id with value of (title, genre)
    movies: list of movies to extract genres.

  Returns:
    movie_genres: list of genres of all input movies.
  """
  movie_genres = []
  for movie in movies:
    if not movies_dict[movie.movie_id].genres:
      continue
    genres = [
        tf.compat.as_bytes(genre)
        for genre in movies_dict[movie.movie_id].genres.split("|")
    ]
    movie_genres.extend(genres)

  return movie_genres

def _pad_or_truncate_movie_feature(feature, max_len, pad_value):
    feature.extend([pad_value for _ in range(max_len - len(feature))])
    return feature[:max_len]

def generate_examples_from_single_timeline(timeline,
                                           movies_dict,
                                           max_context_len=100,
                                           max_context_movie_genre_len=320):
    """Generate TF examples from a single user timeline."""
    examples = []
    for label_idx in range(1, len(timeline)):
        start_idx = max(0, label_idx - max_context_len)
        context = timeline[start_idx:label_idx]
        while len(context) < max_context_len:
            context.append(MovieInfo())
        label_movie_id = int(timeline[label_idx].movie_id)
        context_movie_id = [int(movie.movie_id) for movie in context]
        context_movie_rating = [movie.rating for movie in context]
        context_movie_year = generate_feature_of_movie_years(movies_dict, context)
        context_movie_genres = generate_movie_genres(movies_dict, context)
        context_movie_genres = _pad_or_truncate_movie_feature(context_movie_genres, max_context_movie_genre_len, tf.compat.as_bytes(UNKNOWN_STR))
        feature = {
            "context_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=context_movie_id)),
            "context_movie_rating": tf.train.Feature(float_list=tf.train.FloatList(value=context_movie_rating)),
            "context_movie_genre": tf.train.Feature(bytes_list=tf.train.BytesList(value=context_movie_genres)),
            "context_movie_year": tf.train.Feature(int64_list=tf.train.Int64List(value=context_movie_year)),
            "label_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_movie_id]))
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)
    return examples

def generate_examples_from_timelines(timelines,
                                     movies_df,
                                     min_timeline_len=3,
                                     max_context_len=100,
                                     max_context_movie_genre_len=320,
                                     train_data_fraction=0.9,
                                     random_seed=None,
                                     shuffle=True):
    """Convert user timelines to tf examples.
    """
    logging.info("Generating examples from timelines...")
    examples = []
    movies_dict = generate_movies_dict(movies_df)

    for timeline in tqdm(timelines.values(), desc="Processing timelines"):
        if len(timeline) < min_timeline_len:
            continue
        single_timeline_examples = generate_examples_from_single_timeline(
            timeline=timeline,
            movies_dict=movies_dict,
            max_context_len=max_context_len,
            max_context_movie_genre_len=max_context_movie_genre_len
        )
        examples.extend(single_timeline_examples)

    logging.info(f"Total examples generated: {len(examples)}")

    if shuffle:
        logging.info("Shuffling examples...")
        random.seed(random_seed)
        random.shuffle(examples)

    last_train_index = round(len(examples) * train_data_fraction)
    train_examples = examples[:last_train_index]
    test_examples = examples[last_train_index:]

    logging.info(f"Train examples: {len(train_examples)}, Test examples: {len(test_examples)}")
    return train_examples, test_examples

def generate_movie_feature_vocabs(movies_df, movie_counts, movie_sum_of_rating):
    """Generate vocabularies for movie features."""
    movie_vocab = []
    movie_genre_counter = collections.Counter()
    movie_year_counter = collections.Counter()
    global_avg_rating = movie_sum_of_rating.total() / movie_counts.total()

    for movie_id, title, genres in movies_df.values:
        count = movie_counts.get(movie_id)
        avg_rating = global_avg_rating if (count and count > 0) else 0
        avg_rating = movie_sum_of_rating.get(movie_id) / count if count and count > 0 else global_avg_rating
        genre_list = genres.split("|")
        row = {"id": movie_id, "title": title, "genres": genre_list, "count": count, "avg_rating": "{:.4f}".format(avg_rating)}
        movie_vocab.append(row)
        year = extract_year_from_title(title)
        movie_year_counter[year] += 1
        for genre in genre_list:
            movie_genre_counter[genre] += 1

    movie_year_vocab = [0] + [x for x, _ in movie_year_counter.most_common(MAXIUM_GENRES_NUM)]
    movie_genre_vocab = [UNKNOWN_STR] + [x for x, _ in movie_genre_counter.most_common(MAXIUM_GENRES_NUM)]
    return movie_vocab, movie_year_vocab, movie_genre_vocab, movie_genre_counter

def write_tfrecords(tf_examples, filename):
    """Writes tf examples to tfrecord file, and returns the count."""
    logging.info(f"Writing TFRecords to {filename}")
    with tf.io.TFRecordWriter(filename) as file_writer:
        for example in tqdm(tf_examples, desc="Writing examples"):
            file_writer.write(example.SerializeToString())
    return len(tf_examples)

def write_vocab_json(vocab, filename):
    """Write generated movie vocabulary to specified file."""
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(vocab, jsonfile, indent=2)

def write_vocab_txt(vocab, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in vocab:
            f.write(str(item) + "\n")

def generate_datasets(extracted_data_dir,
                      output_dir,
                      min_timeline_length,
                      max_context_length,
                      max_context_movie_genre_length,
                      min_rating=None,
                      build_vocabs=True,
                      train_data_fraction=0.9):
    """Generates train and test datasets as TFRecord, and returns stats."""
    logging.info(f"Reading data from {extracted_data_dir}")
    ratings_df, movies_df = read_data(extracted_data_dir, min_rating=min_rating)

    logging.info("Generating movie rating user timelines")
    timelines, movie_counts, movie_sum_of_rating = convert_to_timelines(ratings_df)

    logging.info("Generating train and test examples")
    train_examples, test_examples = generate_examples_from_timelines(
        timelines=timelines,
        movies_df=movies_df,
        min_timeline_len=min_timeline_length,
        max_context_len=max_context_length,
        max_context_movie_genre_len=max_context_movie_genre_length,
        train_data_fraction=train_data_fraction
    )

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    train_filename = OUTPUT_TRAINING_DATA_FILENAME.format(content_name=FLAGS.content_name)
    test_filename = OUTPUT_TESTING_DATA_FILENAME.format(content_name=FLAGS.content_name)

    logging.info("Writing generated training examples")
    train_file = os.path.join(output_dir, train_filename)
    train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)

    logging.info("Writing generated testing examples")
    test_file = os.path.join(output_dir, test_filename)
    test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)

    stats = {
        "train_size": train_size,
        "test_size": test_size,
        "train_file": train_file,
        "test_file": test_file,
    }

    if build_vocabs:
        logging.info("Generating movie feature vocabularies")
        movie_vocab, movie_year_vocab, movie_genre_vocab, movie_genre_counter = generate_movie_feature_vocabs(
            movies_df, movie_counts, movie_sum_of_rating)

        vocab_file = os.path.join(output_dir, OUTPUT_MOVIE_VOCAB_FILENAME.format(content_name=FLAGS.content_name))
        write_vocab_json(movie_vocab, vocab_file)
        stats["vocab_file"] = vocab_file
    # sort coutner before export
        vocab_json_filename = os.path.join(output_dir, OUTPUT_GENRE_VOCAB_FILENAME.format(content_name=FLAGS.content_name))
        write_vocab_json(sorted(movie_genre_counter.items(), key=lambda pair: pair[1], reverse=True), filename=vocab_json_filename)
        stats["genre_vocab_file"] = vocab_json_filename
        year_vocab_file = os.path.join(output_dir, OUTPUT_MOVIE_YEAR_VOCAB_FILENAME.format(content_name=FLAGS.content_name))
        write_vocab_txt(movie_year_vocab, year_vocab_file)
        stats["year_vocab_file"] = year_vocab_file

        genre_vocab_file = os.path.join(output_dir, OUTPUT_MOVIE_GENRE_VOCAB_FILENAME.format(content_name=FLAGS.content_name))
        write_vocab_txt(movie_genre_vocab, genre_vocab_file)
        stats["genre_vocab_file"] = genre_vocab_file

    return stats

def main(_):
    logging.info("Starting data preparation process")
    stats = generate_datasets(
        FLAGS.extracted_data_dir,
        FLAGS.output_dir,
        FLAGS.min_timeline_length,
        FLAGS.max_context_length,
        FLAGS.max_context_movie_genre_length,
        FLAGS.min_rating,
        FLAGS.build_vocabs,
        FLAGS.train_data_fraction
    )

    logging.info("Data preparation completed. Stats:")
    for key, value in stats.items():
        logging.info(f"{key}: {value}")

if __name__ == "__main__":
    define_flags()
    app.run(main)