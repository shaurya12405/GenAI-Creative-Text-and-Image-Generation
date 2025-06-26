import requests
from bs4 import BeautifulSoup
import json
import re
from google.colab import files

result=[]
errors=[]
Total_errors=[]
def web_scrape1(url, i, ind):
# Send a GET request to the webpage
  response = requests.get(url)

# Check if the request was successful
  if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title
    title = soup.find('h1').text.strip() if soup.find('h1') else "Title not found"

    # Extract the author
    author_div = soup.find('div', attrs={"data-byline-author": True})
    author = author_div.text.strip() if author_div else "Author not found"

    # Extract the poem text
    main_poem = soup.find('div', id='block-stanza-content')
    poem_lines = main_poem.find_all('span', class_='long-line') or main_poem.find_all('pre') or main_poem.find_all('p')
    if poem_lines:
      poem_text = "\n".join([line.text.strip() for line in poem_lines])

    else:
      print("Poem not found")
      err_info = {
          "Page": i,
          "URL": url
      }
      errors.append(err_info)
      Total_errors.append(ind)



    # Structure the poem data
    poem_data = {

        "poem": poem_text
    }

    # Print the poem details to the console

    #print(poem_data["poem"])
    #result.append({"Poem": poem_data["poem"]})
    return poem_data

  else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")



def web_scrape2(url, i):

  res= requests.get(url) # Here, we sent an http request for this website
  if res.status_code==200:
    # Checking if the http request has been completed
    soup= BeautifulSoup(res.content, 'html.parser') # parsing the content of the page
    poems=soup.select('tr') # Selecting all 'tr' elements with select method

    if not poems:
            print("No tr found with the specified class.")
            return
    for ind, poem in enumerate(poems):
      title_cell = poem.find('td', class_="views-field-title") # Finding the title element inside a particular poem, by specifying a class particular to poem titles
      #Finding the poet name element inside a particular poem, by specifying a class particular to poet names
      poet_cell = poem.find('td', class_="views-field-field-author")
      #Finding the date element inside a particular poem, by specifying a class particular to dates
      date_cell = poem.find('td', class_="views-field-field-date-published")
      if title_cell and poet_cell and date_cell:

                title = title_cell.find('a').text.strip() #Finding title text inside 'a', and stripping off any whitespaces
                poem_url = "https://poets.org" + title_cell.select('a')[0].get('href') # Adding link to a particular poem, stored in href attribute of its 'a' tag, by get method
                poet_link = poet_cell.find('a')
                poet = poet_link.text.strip() if poet_link else "N/A" # checking if poet's name is given or not
                time_tag = date_cell.find('time')
                date = time_tag.text.strip() if time_tag else "N/A"
                print(f"\033[1mTitle\033[0m: {title}, Poem URL: {poem_url}, Poet: {poet}, Date: {date}")

                poem_data = {
                              "Title": title,
                              "Author": poet,
                              "Date": date,
                              "Poem": web_scrape1(poem_url, i, ind)
                            }
                result.append([poem_data]) # Adding the poem data in result list
                #web_scrape1(poem_url, i, ind)

  else:
        print(f"Failed to retrieve page: {res.status_code}")

# Enter the range between 1 and 789
for i in range(101):
  print(f"Page : {i}")
  web_scrape2("https://poets.org/poems? "+f"page={i}", i)

print(str(len(Total_errors)) + " errors occured")
print(errors)


with open("merged_file.json", "w") as outfile:
    json.dump(result, outfile)
files.download("mg.json")


# Load raw data from the JSON file
with open("/content/dataset.json", "r") as f:
    data = json.load(f)
print(type(data))
print(data[:5])
