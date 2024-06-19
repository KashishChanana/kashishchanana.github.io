AUTHOR = 'Kashish Chanana'
SITENAME = "Hitchhiker's Guide To AI"
SITEURL = ""
SITELOGO = 'images/test/ai.png'

# SITETITLE = "Hitchhiker's \n Guide To AI" 
SITESUBTITLE = 'Honestly started as a note-taking exercise!'

PATH = "content"

TIMEZONE = 'America/Los_Angeles'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

THEME = 'themes/Flex'

SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.6,
        'indexes': 0.6,
        'pages': 0.5,
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly',
    }
}

# Add a link to your social media accounts
SOCIAL = (
    ('github', 'https://github.com/KashishChanana'),
    ('linkedin','https://www.linkedin.com/in/kashishchanana/'),
    ('twitter','https://twitter.com/chankashish'),
    ('envelope', 'mailto:chananakashish1998@gmail.com'),
)

STATIC_PATHS = ['images', 'extra']

# Main Menu Items
MAIN_MENU = True
MENUITEMS = (('Archives', '/archives'),('Categories', '/categories'),('Tags', '/tags'))

# Code highlighting the theme
PYGMENTS_STYLE = 'friendly'

# Blogroll
LINKS = (
    # ("Pelican", "https://getpelican.com/"),
    # ("Python.org", "https://www.python.org/"),
    # ("Jinja2", "https://palletsprojects.com/p/jinja/"),
    # ("You can modify those links in your config file", "#"),
)


DEFAULT_PAGINATION = 10

SITE_LICENSE = """
&copy; Copyright 2024 by Kashish Chanana.
"""
# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

COPYRIGHT_NAME = AUTHOR
COPYRIGHT_YEAR = 2024