from distutils.core import setup

setup(
    name='genbot',  # How you named your package folder (MyLib)
    packages=['genbot'],  # Chose the same as "name"
    version='0.1',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='A machine learning library for creating practical chatbots through labelled conversation data',
    # Give a short description about your library
    author='Bennie',
    author_email='bennie.v.e@live.co.za',
    url='https://github.com/user/reponame',  # Provide either the link to your github or to your website
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['pytorch', 'chatbot', 'machine learning', 'gpt', 'transformers'],
    # Keywords that define your package best
    install_requires=[
        'numpy~=1.24',
        'pandas~=2.0',
        'torch~=2.0',
        'transformers~=4.28',
        'torchmetrics~=0.11.4',
        'inquirer~=3.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
