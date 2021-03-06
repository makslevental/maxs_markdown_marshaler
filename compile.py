#!/usr/bin/python3
# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
import sys, os, time
import markdown
from tkinter.filedialog import askopenfilename
from tkinter import Tk, Button, Radiobutton, StringVar
from tkinter.font import Font
from selenium import webdriver
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

default_extensions = ['markdown.extensions.extra','markdown.extensions.toc', 'markdown.extensions.smarty', 'markdown.extensions.nl2br', 'markdown.extensions.urlize', 'markdown.extensions.Highlighting', 'markdown.extensions.Strikethrough', 'markdown.extensions.markdown_checklist', 'markdown.extensions.superscript', 'markdown.extensions.subscript', 'markdown.extensions.mathjax']


class GUI:
    
    def __init__(self):
        self.app = Tk()
        self.app.title('Bare MarkDown')
        self.app.resizable(width=False, height=False)
#         self.board = Board()
        self.font = Font(family="Helvetica", size=12)
        self.markdown_file_button = Button(self.app, text='Markdown File', command=self.pick_file, font=self.font)
        self.markdown_file_button.grid(row=0,column=0, columnspan=2)
        self.html_template_file_button = Button(self.app, text='HTML Template', command=self.pick_html_template, font=self.font)
        self.html_template_file_button.grid(row=1,column=0,columnspan=2)

        self.v = StringVar()
        self.v.set("Chrome") # initialize
        Radiobutton(self.app, text="Chrome", variable=self.v, value="Chrome").grid(row=2, column=0)        
        Radiobutton(self.app, text="Firefox", variable=self.v, value="Firefox").grid(row=2, column=1)
        self.start_button = Button(self.app, text='Start', command=self.start, font=self.font)
        self.start_button.grid(row=3,column=0, columnspan=4)
    
    def pick_file(self):
        self.markdown_file_path = askopenfilename()
        self.markdown_file_button.config(text=os.path.basename(self.markdown_file_path))
        self.html_file_path = os.path.dirname(self.markdown_file_path)+os.sep+os.path.splitext(os.path.basename(self.markdown_file_path))[0]+'.html'
        
    def pick_html_template(self):
        self.html_template_file_path = askopenfilename()
        with open(self.html_template_file_path) as html:
            self.html = html.read()
        self.html_template_file_button.config(text=os.path.basename(self.html_template_file_path))
    
    def start(self):
        
        if self.start_button.config('text')[-1] == 'Start':
            self.start_button.config(text='Stop')
            self.write_to_html()
            if self.v.get() == "Chrome":
                self.driver = webdriver.Chrome()
            else:
                self.driver = webdriver.Firefox()
                
            self.driver.get('file:///'+self.html_file_path)
            thr = threading.Thread(target=self.watch_file)
            thr.start()
        else:
            self.driver.close()
            sys.exit()
    
    def write_to_html(self):
        text = open(self.markdown_file_path,"r").read()
        markdown_html = markdown.markdown(text, default_extensions)
        html = self.html.format(html=markdown_html)
        tf = open(self.html_file_path,'w+b')
        tf.write(html.encode())
        tf.flush()
        tf.close()
        
    def watch_file(self):
        path = os.path.dirname(self.html_file_path)
        event_handler = self.MyHandler(self.driver, self.write_to_html, self.markdown_file_path)
        observer = Observer()
        observer.schedule(event_handler, path=path, recursive=False)
        observer.start()
        
    class MyHandler(FileSystemEventHandler):
        def __init__(self,driver, render_fn, fp):
            super().__init__()
            self.fp = fp
            self.driver = driver
            self.render_fn = render_fn
            
        def on_modified(self, event):
            if event.src_path == self.fp:
                self.render_fn()
                self.driver.refresh()
                for i in range(10):
                    time.sleep(1)
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight+100);")
        
    def mainloop(self):
        self.app.mainloop()

if __name__ == '__main__':
    GUI().mainloop()
    #driver = webdriver.Firefox()
    #driver.get('http://lkqpickyourpart.com/locations')
    #print(driver.execute_script('return window.maxs_markers'))
