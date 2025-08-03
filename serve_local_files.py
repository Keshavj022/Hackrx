#!/usr/bin/env python3
"""Simple file server to serve documents from data directory"""

import http.server
import socketserver
import os
from threading import Thread
import time

class FileServer:
    def __init__(self, directory="data", port=8001):
        self.directory = directory
        self.port = port
        self.httpd = None
        
    def start(self):
        """Start the file server"""
        os.chdir(self.directory)
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", self.port), handler)
        
        print(f"Serving files from '{self.directory}' directory at http://localhost:{self.port}")
        print("Available files:")
        for file in os.listdir("."):
            if file.endswith(('.pdf', '.docx', '.txt')):
                print(f"  - http://localhost:{self.port}/{file}")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping file server...")
            self.httpd.shutdown()

if __name__ == "__main__":
    server = FileServer()
    server.start()