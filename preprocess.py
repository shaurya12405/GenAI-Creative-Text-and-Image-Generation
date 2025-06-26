import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Process each poem in the nested structure
cleansed_data_list = []

for entry in data:
    # Each entry is a list containing a dictionary
    if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], dict):
        poem_data = entry[0]  # Access the dictionary inside the list

        # Extract fields safely
        title = poem_data.get("Title", "Untitled")
        author = poem_data.get("Author", "Unknown")
        poem_text = poem_data.get("Poem", {}).get("poem", "")

        # Cleansing process for the poem text
        cleaned_poem = re.sub(r"[^\w\s,.'-]", "", poem_text)  # Removes special characters except basic ones
        cleaned_poem = re.sub(r"\s+", " ", cleaned_poem)  # Replaces multiple spaces with a single space
        cleaned_poem = cleaned_poem.replace(" .", ".").replace(" ,", ",")  # Fix misplaced spaces before punctuation

        # Preserving line breaks
        lines = [line.strip() for line in cleaned_poem.split("\n") if line.strip()]  # Remove empty lines
        cleaned_poem = "\n".join(lines)

        # Append the cleansed data
        cleansed_data_list.append({
            "title": title,
            "author": author,
            "poem": cleaned_poem,
            "date": poem_data.get("Date", "Unknown")
        })

# Print the cleansed poems
for cleansed_data in cleansed_data_list:
    print("Cleansed Poem:\n")
    print(f"Title: {cleansed_data['title']}")
    print(f"Author: {cleansed_data['author']}")
    print(f"Date: {cleansed_data['date']}")

    print(cleansed_data['poem'])

with open("cleansed1.txt", "w") as f:
    json.dump(cleansed_data_list, f)

from google.colab import files

files.download('cleansed1.txt')


import json

# Load the first JSON file
with open("/content/dataset (2).json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# Load the second JSON file
with open("/content/dataset.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# Merge the files based on their structure
if isinstance(data1, list) and isinstance(data2, list):
    # Combine the two lists
    merged_data = data1 + data2
elif isinstance(data1, dict) and isinstance(data2, dict):
    # Merge two dictionaries
    merged_data = {**data1, **data2}
else:
    raise ValueError("The JSON files have incompatible structures.")

# Save the merged data to a new file
with open("merged_file.json", "w", encoding="utf-8") as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

print("Files merged successfully!")





file_path = '/content/dataset.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    print(data[:5])
    print(type(data))
    print(len(data))
    print(type(data[0]))
    print(data[0])
    print(type(data[0][0]))
    print(data[0][0])

# Flattening the data
flattened_data = []
for item in data:
    for poem in item:
        poem_data = {
            "Title": poem.get("Title", "Unknown"),
            "Author": poem.get("Author", "Unknown"),
            "Date": poem.get("Date", "Unknown"),
            "Poem": poem.get("Poem", {}).get("poem", ""),
            "Poem_Length": len(poem.get("Poem", {}).get("poem", "").split())
        }
        flattened_data.append(poem_data)

df = pd.DataFrame(flattened_data)
df.head(10)



# Inspect the DataFrame
df.info()  # Check column types and non-null counts
df.describe(include="all")  # Summary statistics
df.isnull().sum()  # Check for missing values


# title date vs count()
from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  counted = (series['Date']
                .value_counts()
              .reset_index(name='counts')
              .rename({'index': 'Date'}, axis=1)
              .sort_values('Date', ascending=True))
  xs = counted['Date']
  ys = counted['counts']
  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = df.sort_values('Date', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Date')
_ = plt.ylabel('count()')

# title Date vs Poem_Length

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['Date']
  ys = series['Poem_Length']

  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = df.sort_values('Date', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Date')
_ = plt.ylabel('Poem_Length')

# Count the number of poems per author
author_counts = df["Author"].value_counts()

# Plot the top 10 authors with the most poems
plt.figure(figsize=(10, 6))
author_counts.head(10).plot(kind="bar", color="teal")
plt.title("Top 10 Authors with the Most Poems", fontsize=16)
plt.xlabel("Author", fontsize=12)
plt.ylabel("Number of Poems", fontsize=12)
plt.xticks(rotation=45)
plt.show()


# distribution of publication years
plt.figure(figsize=(10, 6))
flattened_data['Date'] = pd.to_numeric(flattened_data['Date'], errors='coerce')  # Convert to numeric for sorting
flattened_data['Date'].dropna().astype(int).plot(kind='hist', bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Publication Years')
plt.xlabel('Year')
plt.ylabel('Number of Poems')
plt.grid(axis='y')
plt.show()

# Distribution of poem lengths
plt.figure(figsize=(10, 6))
sns.histplot(df["Poem_Length"], bins=30, kde=True, color="blue")
plt.title("Distribution of Poem Lengths", fontsize=16)
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()


# Analyze poem lengths (words and lines)
# Convert flattened_data to a DataFrame if it's not already
import pandas as pd
flattened_data = pd.DataFrame(flattened_data) # convert flattened_data to a DataFrame if it's a list

flattened_data['Word Count'] = flattened_data['Poem'].apply(lambda x: len(str(x).split()))
flattened_data['Line Count'] = flattened_data['Poem'].apply(lambda x: len(str(x).splitlines()))

print("\n5. Average poem length (words):", flattened_data['Word Count'].mean())
print("6. Average poem length (lines):", flattened_data['Line Count'].mean())

# Display the first few rows
print("Sample Data:")
print(flattened_data.head())
# Display the poems with the longest and shortest word counts
longest_poem = flattened_data.loc[flattened_data['Word Count'].idxmax()]
shortest_poem = flattened_data.loc[flattened_data['Word Count'].idxmin()]

print("\n7. Longest Poem:")
print(longest_poem[['Title', 'Author', 'Word Count']])

print("\n8. Shortest Poem:")
print(shortest_poem[['Title', 'Author', 'Word Count']])

# Save the processed DataFrame as a CSV file
df.to_csv("processed_poems.csv", index=False)
from google.colab import files
files.download("processed_poems.csv")

