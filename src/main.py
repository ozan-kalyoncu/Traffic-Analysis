#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function

import requests
from urllib.parse import urlencode
from googleapiclient.discovery import build 
from google.oauth2 import service_account
import datetime
import pandas as pd
import time
import sched
import schedule
import socket

class DataSpreadSheets:
    def __init__(self, origin, destination, spreadsheetId) -> None:
        
        # API Key and Sheets ID
        self.api_key = "AIzaSyCJwXP-NsoF55sozWXRHaJYq37uhGG6w40"
        self.spreadsheetId = spreadsheetId

        # Params
        self.paramsOrigin = {
            "address": origin,
            "key": self.api_key
        }

        self.paramsDestination = {
            "address": destination,
            "key": self.api_key
        }

        # Initial Creds
        self.creds = None


        # Inner Scope Modules
        self.datetime = datetime.datetime
        self.pd = pd
        self.scheduler = sched.scheduler(time.time, time.sleep)
        

        # Maps API Endpoint    
        self.endpoint = "https://maps.googleapis.com/maps/api/geocode/{0}".format("json")  #Base url
        
        # Distance Matrix Variables
        self.matrixBaseUrl = f"https://maps.googleapis.com/maps/api/distancematrix/json"

        self.titles = [["Hour", "ID", "Origin Main City", "Origin City", "Distance Forward", "Duration Forward", "Avg Speed Forward", "Dest City", "Distance Backwards", "Duration Backwards", "Avg Speed Backward"]]

        # Inner Database
        self.db = pd.DataFrame(columns = self.titles)

    def main(self):

        self.setCreds()

        try:
            self.writeSpreadSheet()
            schedule.every(60).minutes.do(self.writeSpreadSheet) 
            
            while True:
                schedule.run_pending()
                time.sleep(1)
    
        except socket.timeout:
            print('Socket Network Connection Error')        
       
    def writeSpreadSheet(self):
        print('Runned')
        results, timeline = self.build_distance_matrix()

        # Store Inner DataFrame
        matrixDb = self.pd.DataFrame(results, columns= self.titles)
        self.db = self.pd.concat([matrixDb, self.db])

        self.updateSpreadSheet(results, timeline)

    def updateSpreadSheet(self, results, timeline):

        day = timeline.date().strftime("%A")
        range = f"{day}!A1:Z"

        bodyUpdate = {
            'values': self.titles + results
        }

        bodyAppend = {
            'values': results
        }

        sheetUpdate = self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheetId, range=range,
            valueInputOption='USER_ENTERED', body=bodyUpdate
        )

        sheetAppend = self.service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheetId, range=range,
            valueInputOption='USER_ENTERED', body=bodyAppend
        )

        sheetRead = self.service.spreadsheets().values().get(
            spreadsheetId= self.spreadsheetId, range=range
        ).execute()

        rows = sheetRead.get('values', [])
        if bool(rows) == False:
            sheetUpdate.execute()
        else:
            sheetAppend.execute()

    def setCreds(self):

        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

        SERVICE_ACCOUNT_FILE = './keys.json'

        self.creds = service_account.Credentials.from_service_account_file( SERVICE_ACCOUNT_FILE, scopes=SCOPES )

        self.service = build('sheets', 'v4', credentials=self.creds)

    def build_distance_matrix(self):

        now = self.datetime.now()

        hour = f"{now.hour}.{now.minute}"
        
        originCords, originCity, mainOriginCity = self.extract_latlong_city_maincity(self.paramsOrigin)

        destCords, destCity, mainDestCity = self.extract_latlong_city_maincity(self.paramsDestination)

        result = []

        distOriginDest, durationOriginDest, avgSpeedOriginDest = self.distance_matrix(originCords, destCords)
        
        distDestOrigin, durationDestOrigin, avgSpeedDestOrigin = self.distance_matrix(destCords, originCords)
        
        result.append([hour, 1, mainOriginCity, originCity, distOriginDest, durationOriginDest, avgSpeedOriginDest, destCity, distDestOrigin, durationDestOrigin, avgSpeedDestOrigin])
        
        return result, now

    def distance_matrix(self, origin, destination):
        
        self.distMatrixParams = {
            "origins": origin,
            "destinations": destination,
            "departure_time": "now",
            "key": self.api_key
        }

        data = self.getRequest(self.distMatrixParams, self.matrixBaseUrl).json()
        
        distance_meter = data['rows'][0]['elements'][0]['distance'].get("value")
        distance = float(distance_meter/1000) #meter --> km
        
        duration_second = data['rows'][0]['elements'][0]['duration_in_traffic'].get("value")
        duration = int(duration_second/60)  #second --> minutes
        
        avg_speed = distance / (duration/60) 
        
        return distance, duration, avg_speed
        

    def extract_latlong_city_maincity(self, params):

        request = self.getRequest(params, self.endpoint)

        address_info = request.json()["results"][0]

        city = ""
        main_city = ""

        for dict in address_info["address_components"]:
            if "administrative_area_level_2" in dict["types"]:
                city = dict["short_name"]
            if "administrative_area_level_1" in dict["types"]:
                main_city = dict["short_name"]

        lat = address_info["geometry"]["location"].get("lat")

        lng = address_info["geometry"]["location"].get("lng")


        return f'{lat},{lng}', city, main_city
        
    def getRequest(self, params, baseUrl):
        
        url_params = urlencode(params)

        url = f'{baseUrl}?{url_params}'

        return requests.get(url)

address = [1,"Altunizade, 34662 Üsküdar/İstanbul"]

dest = [2, "YTÜ-Davutpaşa Kampüsü, 34220 Davutpaşa/İstanbul"]

spreadsheet = DataSpreadSheets(origin=address, destination=dest, spreadsheetId="1VbdnVB0V4doRK4TLkeoBKajV9a8N4RcD627I0imiVyU")

spreadsheet.main()
