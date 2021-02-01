import os
from urllib.request import urlretrieve
from selenium import webdriver
DRIVER_PATH = 'C:/Users/Dylan/PycharmProjects/chromedriver.exe'
NUM_IMAGES_TO_DOWNLOAD = 1000

local_image_folder = './lasagna/'
# local_image_folder = './doom/'
if not os.path.exists(local_image_folder):
    os.makedirs(local_image_folder)

web = webdriver.Chrome(executable_path=DRIVER_PATH)
web.get('https://www.google.com/search?q=lasagna&source=lnms&tbm=isch')
# web.get('https://www.google.com/search?q=doom+2016+level&source=lnms&tbm=isch')

urlset = set()
done = False
while not done:
    # get all image thumbnail results
    thumbnail_results = web.find_elements_by_css_selector("img.Q4LuWd")
    number_results = len(thumbnail_results)
    print('Storing ' + str(number_results) + ' more results, set now contains ', str(len(urlset)))
    for img in thumbnail_results:
        if len(urlset) >= NUM_IMAGES_TO_DOWNLOAD:
            done = True
            break
        if img.get_attribute('src') and 'http' in img.get_attribute('src'):
            urlset.add(img.get_attribute('src'))
    # if we get here, we exhausted the page but don't have enough images, so we have to load a new page
    load_more_button = web.find_element_by_css_selector(".mye4qd")
    if load_more_button:
        web.execute_script("document.querySelector('.mye4qd').click();")

# once we're to this point, we have 1000 unique urls, so download them
for image_number, image_url in enumerate(urlset):
    urlretrieve(image_url, local_image_folder + '{:03d}'.format(image_number) + '.png')
