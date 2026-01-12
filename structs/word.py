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
        pt1 = (self.x, self.y + 4)
        pt2 = (self.x + self.w +30, self.y + self.h -4)
        return pt1, pt2

    def __repr__(self):
        return f"{self.word} ({self.confidence}%) -> {self.equivalent}"