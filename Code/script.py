from bs4 import BeautifulSoup

soup = BeautifulSoup(open("/Users/reza/Downloads/cb.html"), "html.parser")
investors = soup.findAll('span', {'class': 'component--field-formatter field-type-identifier-multi'})
companies = soup.findAll('a', {'class': 'component--field-formatter field-type-identifier link-accent ng-star-inserted'})

if len(investors) != 200 or len(companies) != 50:
    print('something is wrong!')
    quit(10)
result = []
for idx, c in enumerate(companies):
    print(c.attrs['title'], end=',')
    for child in investors[idx*4+2].children:
        try:
            print(child.attrs['title'], end=',')
        except:
            pass
    print('\n')