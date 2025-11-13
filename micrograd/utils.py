
# class RNG mimics the random interface in Python, fully deterministic,

class RNG:

    def __init__(self, seed):
        self.state = seed
    
    #The method generates random numbers that are 32-bit unsigned integers.
    def random_u32(self):

        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF
    

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0


    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()
    

    def gen_data(random: RNG, n=100):
        
        # Initialize an empty list to store data points
        pts = []

        for _ in range(n):

            #Generate random x and y coordinates within the range [-2.0, 2.0].
            x = random.uniform(-2.0, 2.0)
            y = random.uniform(-2.0, 2.0)

            label = 0 if x < 0 else 1 if y < 0 else 2

            pts.append(([x, y], label))

            tr = pts[:int(0.8*n)]
            val = pts[int(0.8*n):int(0.9*n)]
            te = pts[int(0.9*n):]

            return tr, val, te