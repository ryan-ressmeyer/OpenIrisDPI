import socket
import time
import json
from dataclasses import dataclass
import math

class Point:
    """A simple 2D point class"""
    def __init__(self, x:float=0, y:float=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)
    
    def copy(self):
        return Point(self.x, self.y)

    def clip(self, minimum, maximum):
        self._x = max(minimum, min(self._x, maximum))
        self._y = max(minimum, min(self._y, maximum))
        return self

    def rotate(self, angle:float):
        R = [[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]]
        x = self._x * R[0][0] + self._y * R[0][1]
        y = self._x * R[1][0] + self._y * R[1][1]
        self._x, self._y = x, y
        return self
    
    def __repr__(self):
        return f'Point({self._x}, {self._y})'
    
@dataclass
class EyeData:
    """A class to hold data from a single eye from OpenIris"""
    frame_number:int
    pupil: Point
    pupil_area:float
    cr: Point
    p4: Point
    cr_error: str
    p4_error: str
    def __init__(self, struct:dict = {}):
        if struct:
            self.frame_number=struct['FrameNumber']
            self.pupil = Point(struct['Pupil']['Center']['X'], struct['Pupil']['Center']['Y'])
            self.pupil_area = struct['Pupil']['Size']['Width'] * struct['Pupil']['Size']['Height']
            if struct['CRs']:
                self.cr = Point(struct['CRs'][0]['X'], struct['CRs'][0]['Y'])
                self.cr_error = ''
            else:
                self.cr = Point(0,0)
                self.cr_error = 'No CRs'
            if len(struct['CRs']) >= 4:
                self.p4 = Point(struct['CRs'][3]['X'], struct['CRs'][3]['Y'])
                self.p4_error = ''
            else:
                self.p4 = Point(0,0)
                self.p4_error = 'No P4'
        else:
            self.frame_number = 0
            self.pupil_area = 0.0
            self.pupil = Point(0,0)
            self.cr = Point(0,0)
            self.p4 = Point(0,0)
            self.cr_error = 'No Data'
            self.p4_error = 'No Data'

    def __repr__(self):
        return f"EyeData({self.frame_number}, Pupil={self.pupil}, Pupil Area={self.pupil_area}, CR={self.cr}, P4={self.p4})"

@dataclass
class EyesData:
    """A class to hold data from both eyes from OpenIris"""
    left: EyeData
    right: EyeData
    error: str
    def __init__(self, struct:dict = {}):
        if struct:
            self.left = EyeData(struct['Left'])
            self.right = EyeData(struct['Right'])
            self.error = ''
        else:
            self.left = EyeData()
            self.right = EyeData()
            self.error = 'No Data'
    
    def __repr__(self):
        return f"EyesData\n\tLeft: {repr(self.left)}\n\tRight: {repr(self.right)}"
    
    def get_error(self, left_p4:bool=True, right_p4:bool=True) -> str:
        if self.error:
            return self.error
        error = ''
        if self.left.cr_error or (self.left.p4_error and left_p4):
            error += f"Left:"
            if self.left.cr_error:
                error += f" {self.left.cr_error}"
            if self.left.p4_error and left_p4:
                error += f" {self.left.p4_error}"
            if self.right.cr_error or (self.right.p4_error and right_p4):
                error += ', '
        if self.right.cr_error or (self.right.p4_error and right_p4):
            error += f"Right:"
            if self.right.cr_error:
                error += f" {self.right.cr_error}"
            if self.right.p4_error and right_p4:
                error += f" {self.right.p4_error}"

        return error

class OpenIrisClient:
    """A simple class to fetch data from OpenIris"""

    def __init__(self, server_address='localhost', port=9003, timeout=1):
        """Initialize the client with the server address, port, and timeout"""
        self.server_address = (server_address, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout) # 200 Hz

    def fetch_data_raw(self, debug=False)->str:
        """Fetch the next processed frame from the server"""
        try:
            self.sock.sendto("WAITFORDATA".encode("utf-8"), self.server_address)
            data = self.sock.recvfrom(8192)  # Adjust the buffer size as needed
            return data[0].decode("utf-8")
        except Exception as e:
            if debug:
                print(f"Error receiving data: {e}")
            return '{}'
        
    def fetch_data_json(self, debug=False)->dict:
        """Fetch the next processed frame from the server as a JSON object"""
        return json.loads(self.fetch_data_raw(debug))
    
    def fetch_data(self, debug=False)->EyesData:
        """Fetch the next processed frame from the server as an EyesData object"""
        return EyesData(self.fetch_data_json(debug))
    
    def __enter__(self):
        self.sock.connect(self.server_address)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()
        if exc_type:
            print(f"Exception: {exc_type} {exc_value}")
            return False
        return True
    
if __name__ == "__main__":
    ip = input("Enter the IP address of the OpenIris server (default=localhost): ")
    if not ip:
        ip = 'localhost'
    with OpenIrisClient(ip) as client:
        while True:
            data = client.fetch_data(True)
            if data is not None:
                print(f"Received data: {data}")
            else:
                break
            time.sleep(1)
    
