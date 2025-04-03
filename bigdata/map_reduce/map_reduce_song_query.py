# Required packages
from pymongo import MongoClient
import pandas as pd
import re
from mrjob.job import MRJob
from mrjob.step import MRStep
import re

# 1.1: Establish a connection to the MongoDB database and query the `song` collection
def song_query():
    # Connect to the `assignment1` database and `song` collection
    client = MongoClient('localhost:27017')
    db = client['assignment1']
    song = db['song']
    
    # Query for desired fields in `song` collection
    result = song.find(
        {},
        {
            '_id': 0,
            'Artist' : 1,
            'Year': 1,
            'Sales': 1
        }
    )
    for field in result:
        # Define the variables to take the value of each field
        artist = field['Artist']
        year = field['Year']
        sales = field['Sales']
        # Print the value in strings
        try:
            print(
                # Print to match the question format
                # Connect each string with seperators, make them inside angle brackets
                '<' + str(artist) + ', ' + str(year) + ', ' + str(sales) + '>'
            )
        # Return error in Exception
        except Exception as e:
            print(
                f"{e} field doesn't exist"
            )

# 1.2: Calculate the sum of sales for each artist and year
class sales_annual_sum(MRJob):
    # Define steps
    def steps(self):
        return[
            MRStep(
                mapper = self.mapper_extract_data,
                reducer = self.reducer_sum_sales
            )
        ]
    # Define the mapper   
    def mapper_extract_data(self, _, line): 
        # Capture strings: "(artist),(year),(sales)"
        #   - '<' and '>': except the two angle brackets
        #   - Use inside quotes and next to `,` to represent match the format of the data
        #   - Each string group will be matched inside each parentheses '()'.
        #       + ([^,]+): (Artist name) select everything before the first ','.
        #       + (\d{4}): (Year) take the exact next 4 digit
        #       + ([\d.]+): (Sales no) select the sales number 
        re_pattern = re.compile(r"<([^,]+), (\d{4}), ([\d.]+)>")
        # As we have compiled the strings into seperate group pattern
        # Now use search() to scan re.compile object then define each matched pattern group
        details = re_pattern.search(line)
        # For each line
        if details:
            # Match the 1st parenthesized group of the string to be artist name
            artist_name = details.group(1)
            # Match the 2nd parenthesized group of the string to be year
            year = details.group(2)
            # Match the 3rd parenthesized group of the string to be sales number
            # Convert the sales into float, then multiply by 1000 to get the actual sales
            # Finally convert each sales number into integer for consistency and future use
            sales = int(float(details.group(3)) * 1000)
            # Return the result
            yield (artist_name, year), sales
    # Define the reducer
    def reducer_sum_sales(self, artist_year, sales):
        # Combine the number of sales of each artist and year
        yield artist_year, sum(sales)

# 2.1: Find the top selling artist for each year
class yearly_top_sales(MRJob):
    # Define steps
    def steps(self):
        return[
            MRStep(
                mapper = self.mapper_extract_data,
                reducer = self.reducer_yearly_sales
            ),
            MRStep(
                reducer = self.reducer_sort_year
            )
        ]
    # Define the mapper  
    def mapper_extract_data(self, _, line):
        # Apply similar methods from task1_2
        #   - Group 1 (artist name) & 2 (year): Use `\"` to match everything `(.*)` inside quotation marks " " 
        #   - Group 3 (sales no): Use \t to match the horizontal space and ([\d]+) to match any digit found
        re_pattern = re.compile(r"\[\"(.*)\", \"(.*)\"]\t([\d]+)")
        details = re_pattern.search(line)
        # For each line
        if details:
            artist = details.group(1)
            year = details.group(2)
            sales = int(details.group(3))
            yield year, (artist, sales)
        
    # Define 1st reducer    
    def reducer_yearly_sales(self, year, artist_sales):
        # Find max sales each year
        top_sales = max(artist_sales, key=lambda x: x[1]) # Use second element `sales` of (artist, sales) as the key for sorted()
        yield None, (year, top_sales)
        
    # Define 2nd reducer
    def reducer_sort_year(self, _, year_top_sales):
        # Sorting years in the descending order
        sorted_year = sorted(
            year_top_sales, 
            key=lambda x: x[0],  # Use first element `year` of (year, top_sales) as the key for sorted()
            reverse=True         # Descending order
        )
        for year, top_sales in sorted_year:
            yield year, top_sales
    
# 2.2: Find the top 5 selling artists of all time            
class top_5_sales_artists(MRJob):
    def steps(self):
        return[
            MRStep(
                mapper = self.mapper_extract_data,
                combiner = self.combiner_alltime_sales,
                reducer = self.reducer_alltime_sales
            ),
            MRStep(
                reducer = self.reducer_top_five
            )
        ]
        
    def mapper_extract_data(self, _, line):
        # Apply similar methods from task1_2
        #   - Group 1 (artist name) & 2 (year): Use `\"` to match everything `(.*)` inside quotation marks " " 
        #   - Group 3 (sales no): Use \t to match the horizontal space and ([\d]+) to match any digit found
        re_pattern = re.compile(r"\[\"(.*)\", \"(.*)\"]\t([\d]+)")
        details = re_pattern.search(line)
        # For each line
        if details:
            artist = details.group(1)
            sales = int(details.group(3))
            yield artist, sales
    
    def combiner_alltime_sales(self, artist, sales):
        # Combine the number of sales of each artist
        yield artist, sum(sales)
            
    def reducer_alltime_sales(self, artist, sales):
        # Aggregate the number of sales of each artist across the data
        yield None, (artist, sum(sales))
    
    def reducer_top_five(self, _, artist_all_sales):
        # Find highest sales each year
        top_5_sales = sorted(
            artist_all_sales, 
            key = lambda x: x[1],   # Use second element `sales` of (artist, sales) as the key for sorted()
            reverse = True          # Descending order
            )[:5]                   # Print first 5 items
        for artist, sales in top_5_sales:
            yield artist, sales
      
# 2.3: Find the top 3 selling artists for each decade      
class top3_sales_by_decade(MRJob):
    def steps(self):
        return[
            MRStep(
                mapper  = self.mapper_extract_data,
                combiner = self.combine_alltime_sales,
                reducer = self.reducer_sort_year
            ),
            MRStep(
                reducer = self.reducer_top_three
            )
        ]
    def mapper_extract_data(self, _, line):
        # Similar methods with previous tasks
        re_pattern = re.compile(r"\[\"(.*)\", \"(.*)\"]\t([\d]+)")
        details = re_pattern.search(line)
        # For each line
        if details:
            artist = details.group(1)
            year = int(details.group(2))
            sales = int(details.group(3))
            # Convert years to decades 
            start_decade = (year // 10) * 10 # Decade starting year
            end_decade = start_decade + 9 # Decade ending year
            decade_range = f"{start_decade}-{end_decade}" # Match starting-ending years together
            yield (decade_range, artist), sales

    def combine_alltime_sales(self, decade_artist, sales):
        # Combine the number of sales of each artist
        yield decade_artist, sum(sales)

    def reducer_sort_year(self, decade_artist, sales):
        # Sort years in the descending order
        decade, artist = decade_artist
        yield decade, (artist, sum(sales))   
        
    def reducer_top_three(self, decade, artist_total_sales):
        # Find highest sales each year
        top_three = sorted(
            artist_total_sales, 
            key = lambda x: x[1],   # Use second element `sales` of (artist, sales) as the key for sorted()
            reverse = True          # Descending order
            )[:3]                   # Print first 3 items
        for artist, sales in top_three:
            yield (decade, artist), sales

if __name__ == '__main__':
    # In practice, each class was ran seperately to generate separate outputs
    song_query()
    sales_annual_sum.run()
    yearly_top_sales.run()
    top_5_sales_artists.run()
    top3_sales_by_decade.run()