from typing import List, Tuple, Dict, Any, TypedDict
import random
import numpy as np
import hashlib
import json
try:
    from requests import request as _request

    async def request(**kwargs):
        resp = _request(kwargs.pop('method', 'GET'), kwargs.pop('url'), **kwargs)
        return resp.status_code, resp.text
    
except ImportError:
    from pyodide.http import pyfetch

    async def request(**kwargs):
        if (data := kwargs.pop("data", None)):
            kwargs.update(body=data)
        resp = await pyfetch(**kwargs)
        return resp.status, await resp.string()

class IntermediateIntegersLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, v: int):
        self.logs.append(int(v))
        
    def dump(self):
        max_log_len = 10
        log_len = min(len(self.logs), max_log_len)
        if log_len == 0:
            return []
        step_increment = len(self.logs) / log_len
        return [
            (step, self.logs[step - 1]) 
            for step in [
                int((i + 1) * step_increment)
                for i in range(log_len)
            ]
        ]

def calcSeed(player_id: str, block_id: str, prev_block_id: str, algorithm_id: str, challenge_id: str, difficulty: dict, nonce: int) -> int:
    seed_phrase = ",".join([
        player_id,
        block_id,
        prev_block_id,
        algorithm_id,
        challenge_id,
        str(sorted(difficulty.items())),
        str(nonce)
    ])
    seed = int.from_bytes(hashlib.sha256(seed_phrase.encode()).digest()[-4:], "big")
    return seed

def calcSolutionHash(solution_base64: str):
    return int.from_bytes(hashlib.sha256(solution_base64.encode()).digest()[-4:], "big")

def randomInterpolate(points, min_point):
    min_x, min_y = min_point
    max_x = max(x for x, y in points)
    max_y = max(y for x, y in points)
    
    points = set(tuple(p) for p in points)
    # Add points right on the bounds so we can interpolate across the full x and y range
    if not any(x == min_x for x, y in points):
        points.add((min_x, max_y))
    if not any(y == min_y for x, y in points):
        points.add((max_x, min_y))
    
    # Interpolate a random x, y point
    if len(points) < 2:
        random_x, random_y = list(points)[0]
    elif 0.5 < random.random():
        # random x, interpolate y
        points = sorted(points, key=lambda p: p[0]) # sort by x
        random_x = random.randint(min_x, max_x)
        x_idx = np.searchsorted(
            [x for x, y in points],
            random_x
        )
        if random_x == points[x_idx][0]: # existing point, no need to interpolate
            random_y = points[x_idx][1]
        else:
            random_y = int(np.round(np.interp(
                random_x, 
                [points[x_idx - 1][0], points[x_idx][0]], 
                [points[x_idx - 1][1], points[x_idx][1]]
            )))
    else:
        # random y, interpolate x
        points = sorted(points, key=lambda p: p[1]) # sort by y
        random_y = random.randint(min_y, max_y)
        y_idx = np.searchsorted(
            [y for x, y in points],
            random_y
        )
        if random_y == points[y_idx][1]: # existing point, no need to interpolate
            random_x = points[y_idx][0]
        else:
            random_x = int(np.round(np.interp(
                random_y, 
                [points[y_idx - 1][1], points[y_idx][1]], 
                [points[y_idx - 1][0], points[y_idx][0]]
            )))

    return random_x, random_y

def minJsonDump(o):
    return json.dumps(o, separators=(',', ':'))

class Proof(TypedDict):
    nonce: int
    solution: Any
    intermediate_integers: List[Tuple[int, int]]

class APIException(Exception): pass

class API:
    def __init__(self, api_url: str, api_key: str, player_id: str):
        self.api_url = api_url
        self.api_key = api_key
        self.player_id = player_id

    async def _call(self, query, data=None):
        status_code, body = await request(
            url=f"{self.api_url}/{query}",
            method='POST' if data else 'GET',
            headers={
                'X-Api-Key': self.api_key,
                'Content-Type': 'application/json'
            },
            data=data
        )
        if status_code == 200:
            return json.loads(body)
        else:
            raise APIException(body)

    async def getPlayerSummary(self):
        return await self._call(f"tig/getPlayerSummary?player_id={self.player_id}")

    async def getPlayerBenchmarks(self):
        return await self._call(f"tig/getPlayerBenchmarks?player_id={self.player_id}")

    async def getLatestBlock(self):
        return await self._call("tig/getLatestBlock")

    async def getRoundConfig(self):
        return await self._call("tig/getRoundConfig")

    async def getAlgorithms(self):
        return await self._call(f"tig/getAlgorithms")

    async def getFrontiers(self, challenge_id: str):
        return await self._call(f"tig/getFrontiers?challenge_id={challenge_id}")

    async def submitBenchmark(
        self, 
        player_id: str,
        block_id: str,
        prev_block_id: str,
        algorithm_id: str,
        challenge_id: str,
        difficulty: Dict[str, int],
        nonces: List[int],
        solution_hashes: List[int],
        compute_times: List[int],
      ):
          return await self._call(
            f"player/submitBenchmark", 
            data=minJsonDump({
                'player_id': player_id,
                'block_id': block_id,
                'prev_block_id': prev_block_id,
                'challenge_id': challenge_id,
                'algorithm_id': algorithm_id,
                'difficulty': minJsonDump(difficulty), 
                'nonces': minJsonDump(nonces),
                'solution_hashes': minJsonDump(solution_hashes),
                'compute_times': minJsonDump(compute_times),
            })
          )

    async def submitProofs(
        self, 
        benchmark_id: str,
        proofs: List[Proof]
      ):
          return await self._call(
            f"player/submitProofs", 
            data=minJsonDump({
                'benchmark_id': benchmark_id,
                'proofs': minJsonDump(proofs)
            })
          )
