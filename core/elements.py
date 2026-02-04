
class Word:
    
    def __init__(self,word:str, confidence:int, x:int, y:int, w:int, h:int, id:int):
        self.word = word
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = id
        self.equivalent = None
        
    def get_coor(self):
        return (self.x,self.y,self.w,self.h)
    
    def get_as_point(self):
        pt1 = (self.x, self.y )
        pt2 = (self.x + self.w , self.y + self.h)
        return pt1, pt2

    def __repr__(self):
        return f"{self.word} ({self.confidence}%) -> {self.equivalent}"
    
class Line:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def get_coor(self):
        return (self.x,self.y,self.w,self.h)
    
    def __repr__(self):
        return f"Line at: [x,y,w,h]{self.get_coor()}"
    
import pandas as pd

class Saver:
    def __init__(self, words: list):
        """
        words: Word instance'ları içeren liste
        """
        self.words = words

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "word": [w.word for w in self.words],
            "equviliant": [w.equivalent for w in self.words]
        }
        return pd.DataFrame(data)

    def save(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
