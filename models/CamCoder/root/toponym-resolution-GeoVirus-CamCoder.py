# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 11:42:44 2022
"""
import xml.dom.minidom
import xml.etree.cElementTree as et
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request
import os

import pickle
import codecs
import sqlite3
from genericpath import isfile
from os import listdir
import spacy
import numpy as np
from geopy.distance import great_circle
from keras.models import load_model
from preprocessing import index_to_coord, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1, get_coordinates
from preprocessing import CONTEXT_LENGTH, pad_list, TARGET_LENGTH, UNKNOWN, REVERSE_MAP_2x2
from text2mapVec import text2mapvec

## calculate the key for each annotated location in mlg data patches
def calc_mlg_key(toponym, old_coord):
    toponym = toponym.strip(' ')
    old_coord_split = old_coord.split(',')
    old_lat = old_coord_split[0].strip(' ')
    old_lon = old_coord_split[1].strip(' ')
    return toponym+' '+old_lat+','+old_lon

## calculate the key for each annotated location in GeoVirus
def calc_geovirus_key(name, lat, lon):
    name = name.strip(' ')
    if (lat != None) & (lon != None):
        lat = lat.strip(' ')
        lon = lon.strip(' ')
        return name+' '+lat+','+lon

## calculate the median error distances
def calc_median_error_distance(new_coord, geocoded_coordinates_list):
    errors = []
    new_coord = new_coord.split(',')
    lat = float(new_coord[0].strip(' '))
    lon = float(new_coord[1].strip(' '))
    for (geoparsed_lat, geoparsed_lon) in geocoded_coordinates_list:
        if (geoparsed_lat, geoparsed_lon) != (None, None):
            errors.append(great_circle((float(lat),float(lon)), (float(geoparsed_lat), float(geoparsed_lon))).km)
    if len(errors) == 0:
        return None
    else:
        return np.median(errors)

## geoparsing function provided by CamCoder
## modified based on our study
def geoparse(text):
    """
    This function allows one to geoparse text i.e. extract toponyms (place names) and disambiguate to coordinates.
    :param text: to be parsed
    :return: currently only prints results to the screen, feel free to modify to your task
    """
    parsed_locations_info = []
    doc = nlp(text)  # NER with Spacy NER
    for entity in doc.ents:
        if entity.label_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
            name = entity.text if not entity.text.startswith('the') else entity.text[4:].strip()
            start = entity.start_char if not entity.text.startswith('the') else entity.start_char + 4
            end = entity.end_char
            near_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, entity.start - CONTEXT_LENGTH / 2):entity.start]], True, padding) + \
                       pad_list(CONTEXT_LENGTH / 2, [x for x in doc[entity.end: entity.end + CONTEXT_LENGTH / 2]], False, padding)
            far_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, entity.start - CONTEXT_LENGTH):max(0, entity.start - CONTEXT_LENGTH / 2)]], True, padding) + \
                      pad_list(CONTEXT_LENGTH / 2, [x for x in doc[entity.end + CONTEXT_LENGTH / 2: entity.end + CONTEXT_LENGTH]], False, padding)
            map_vector = text2mapvec(doc=near_inp + far_inp, mapping=ENCODING_MAP_1x1, outliers=OUTLIERS_MAP_1x1, polygon_size=1, db=conn, exclude=name)

            context_words, entities_strings = [], []
            target_string = pad_list(TARGET_LENGTH, [x.text.lower() for x in entity], True, u'0')
            target_string = [word_to_index[x] if x in word_to_index else word_to_index[UNKNOWN] for x in target_string]
            for words in [near_inp, far_inp]:
                for word in words:
                    if word.text.lower() in word_to_index:
                        vec = word_to_index[word.text.lower()]
                    else:
                        vec = word_to_index[UNKNOWN]
                    if word.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
                        entities_strings.append(vec)
                        context_words.append(word_to_index[u'0'])
                    elif word.is_alpha and not word.is_stop:
                        context_words.append(vec)
                        entities_strings.append(word_to_index[u'0'])
                    else:
                        context_words.append(word_to_index[u'0'])
                        entities_strings.append(word_to_index[u'0'])

            prediction = model.predict([np.array([context_words]), np.array([context_words]), np.array([entities_strings]),
                                        np.array([entities_strings]), np.array([map_vector]), np.array([target_string])])
            prediction = index_to_coord(REVERSE_MAP_2x2[np.argmax(prediction[0])], 2)
            candidates = get_coordinates(conn, name)

            if len(candidates) == 0:
                parsed_locations_info.append({'name':name, 'start':start+1, 'end': end+1, 'lat': None, 'lon': None,'candidates': candidates})
                continue

            max_pop = candidates[0][2]
            best_candidate = []
            bias = 0.905  # Tweak the parameter depending on the domain you're working with.
            # Less than 0.9 suitable for ambiguous text, more than 0.9 suitable for less ambiguous locations, see paper
            for candidate in candidates:
                err = great_circle(prediction, (float(candidate[0]), float(candidate[1]))).km
                best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * bias, (float(candidate[0]), float(candidate[1]))))
            best_candidate = sorted(best_candidate, key=lambda a: a[0])[0]

            # England,, England,, 51.5,, -0.11,, 669,, 676 || - use evaluation script to test correctness
            parsed_locations_info.append({'name':name, 'start':start+1, 'end': end+1, 'lat': best_candidate[1][0], 'lon': best_candidate[1][1] ,'candidates': candidates})
    return parsed_locations_info

model = load_model("../data/weights")  # weights to be downloaded from Cambridge Uni repo, see GitHub.
nlp = spacy.load(u'en_core_web_lg')  # or spacy.load(u'en') depending on your Spacy Download (simple or full)
conn = sqlite3.connect(u'../data/geonames.db').cursor()  # this DB can be downloaded using the GitHub link
padding = nlp(u"0")[0]  # Do I need to explain? :-)
word_to_index = pickle.load(open(u"data/words2index.pkl",'rb'))  # This is the vocabulary file
word_to_index = dict((key.strip('\r'), value) for (key, value) in word_to_index.items()) # added by Zilong

for word in nlp.Defaults.stop_words:  # This is only necessary if you use the full Spacy English model
    lex = nlp.vocab[word]             # so if you use spacy.load(u'en'), you can comment this out.
    lex.is_stop = True

## read and preprocess GeoVirus
dir_geovirus = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))+'\\data\\evaluation-corpora\\original-datasets\\GeoVirus.xml'
parsed_xml_geovirus = et.parse(dir_geovirus)

source_list = []
text_list = []
locations_list = []

for article in parsed_xml_geovirus.getroot():
    source = article.find('source')
    text = article.find('text')
    locations = article.find('locations')
    location_list = []
    for location in locations:
        name = location.find('name')
        start = location.find('start')
        end = location.find('end')
        lat = location.find('lat')
        lon = location.find('lon')
        page = location.find('page')
        
        page_split = page.text.split('/')
        wikipedia_name = page_split[len(page_split)-1]
        wikipedia_name = wikipedia_name.replace("_"," ")
        wikipedia_name = wikipedia_name.replace("%27","\'")
        
        location_list.append({'name':name.text, 'wikipedia_name':wikipedia_name, 'start':start.text, 'end': end.text, 'lat': lat.text, 'lon': lon.text, 'page': page.text})    
    
    source_list.append(source.text)
    text_list.append(text.text)
    locations_list.append(location_list)
    

df_geovirus = pd.DataFrame({'source' :source_list, 'text': text_list, 'locations': locations_list})

## extract annotated locations from GeoVirus
name_list = []
lat_list = []
lon_list = []
page_list = []
wikipedia_name_list = []

for article in parsed_xml_geovirus.getroot():
    locations = article.find('locations')
    for location in locations:
        name = location.find('name')
        lat = location.find('lat')
        lon = location.find('lon')
        page = location.find('page')
        
        page_split = page.text.split('/')
        wikipedia_name = page_split[len(page_split)-1]
        wikipedia_name = wikipedia_name.replace("_"," ")
        wikipedia_name = wikipedia_name.replace("%27","\'")
        
        name_list.append(name.text)
        lat_list.append(lat.text)
        lon_list.append(lon.text)
        page_list.append(page.text)
        wikipedia_name_list.append(wikipedia_name)

df_geovirus_poi = pd.DataFrame({'name' :name_list, 'wikipedia_name':wikipedia_name_list, 'lat': lat_list, 'lon': lon_list, 'page': page_list})

df_geovirus_poi = df_geovirus_poi.drop_duplicates()

## read GeoVirus data patches
dir_mlg_geovirus = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))+'\\data\\evaluation-corpora\\data-patches\\GeoVirus_patches.tsv'

mlg_geovirus = pd.read_csv(dir_mlg_geovirus,sep = '\t', header = None)
mlg_geovirus = mlg_geovirus.rename(columns = {0:'toponym', 1:'old_coord', 2:'new_coord'})

mlg_geovirus['mlg_key'] = mlg_geovirus.apply(lambda x: calc_mlg_key(x['toponym'],x['old_coord']),axis = 1)

df_geovirus_poi['geovirus_key'] = df_geovirus_poi.apply(lambda x: calc_geovirus_key(x['wikipedia_name'], x['lat'], x['lon']),axis = 1)

## unifying GeoVirus
df_geovirus_poi_unified = pd.merge(df_geovirus_poi, mlg_geovirus, how = 'outer', left_on = 'geovirus_key', right_on = 'mlg_key')

df_geovirus_poi_unified= df_geovirus_poi_unified.dropna(subset=['new_coord'])

## geoparse articles in GeoVirus
df_geovirus['geoparsed_result'] = df_geovirus['text'].apply(lambda text: geoparse(text))
    
df_geovirus_poi_unified['geocoded_coordinates_list'] = df_geovirus_poi_unified['name'].apply(lambda x: [])

for i in range(len(df_geovirus)):
    toponyms = df_geovirus['locations'].iloc[i]
    geoparsed_result = df_geovirus['geoparsed_result'].iloc[i]
    for toponym in toponyms:
        try:
            df_geovirus_poi_toponym_index = df_geovirus_poi_unified[(df_geovirus_poi_unified['name'] == toponym['name']) & (df_geovirus_poi_unified['lat'] == toponym['lat']) & (df_geovirus_poi_unified['lon'] == toponym['lon'])].index[0]
        except IndexError:
            continue ## no coordinate information for this annotated toponym in geovirus
        for geoparsed_toponym in geoparsed_result:
            if (toponym['name'] == geoparsed_toponym['name']):
                if (int(toponym['start']) == int(geoparsed_toponym['start'])) & (int(toponym['end']) == int(geoparsed_toponym['end'])):
                    toponym['geoparsed_lat'] = geoparsed_toponym['lat']
                    toponym['geoparsed_lon'] = geoparsed_toponym['lon']
                    break
        try:
            df_geovirus_poi_unified['geocoded_coordinates_list'][df_geovirus_poi_toponym_index].append((toponym['geoparsed_lat'],toponym['geoparsed_lon']))
        except KeyError:
            continue ## no coordinate information for this annotated toponym in GeoNames

## MdnED calculation
df_geovirus_poi_unified['median_error_distance'] = df_geovirus_poi_unified.apply(lambda x: calc_median_error_distance(x['new_coord'], x['geocoded_coordinates_list']), axis = 1)

## save toponym resolution results
dir_geovirus_results = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))+'\\geoparsed-results'
df_geovirus_poi_unified.to_csv(dir_geovirus_results+'\\geovirus_geocoded_results_CamCoder.csv')