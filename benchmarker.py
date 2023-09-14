import random
import os
import asyncio
import numpy as np
import base64
from uuid import uuid4
from datetime import datetime, timedelta
from utils import API, calcSeed, calcSolutionHash, randomInterpolate, request, minJsonDump
from js import TIG
API_URL = TIG.API_URL
GIT_BRANCH = TIG.GIT_BRANCH


class SimpleBenchmarker:
    def __init__(self, api: API):
        self.api = api
        self.running = True
        self.status = "Initialising"
        self.summary = None
        self.all_algorithms = None
        self.selected_algorithms = {}
        self.recent_benchmarks = None
        self.low_performance = []
        self.latest_block = None
        self.round_config = None
        self.frontiers = None
        self.reset()

    def reset(self):
        self.proofs = []
        self.nonce = 0
        self.num_errors = 0
        self.benchmark_start = None
        self.benchmark_end = None
        self.challenge_id = None
        self.algorithm_id = None
        self.difficulty = None

    def pickChallengeToBenchmark(self):
        # an imbalance penalty is applied to our benchmarker earnings based on 
        # the standard deviation of our earnings across challenges (smaller deviation = smaller penalty)
        # this incentivises us to spread our compute time across challenges
        # pick the challenge we are earning the least to reduce the penalty
        challenge_earnings = {
            challenge_id: 0
            for challenge_id in self.round_config["difficulty_bounds"]
        }
        for row in self.recent_benchmarks["data_rows"]:
            b = dict(zip(self.recent_benchmarks["header_row"], row))
            challenge_earnings[b["challenge_id"]] += b["latest_earnings"]
        min_earnings = min(challenge_earnings.values())
        return random.choice([
            challenge_id
            for challenge_id, earnings in challenge_earnings.items()
            if earnings == min_earnings
        ])

    def pickAlgorithmToBenchmark(self):
        selected = self.selected_algorithms.get(self.challenge_id)
        for row in self.all_algorithms["data_rows"]:
            row = dict(zip(self.all_algorithms["header_row"], row))
            if (
                row["challenge_id"] == self.challenge_id and 
                row["algorithm_id"] == selected and 
                not row["banned"] and
                datetime.now().astimezone() >= datetime.fromisoformat(row["datetime_to_be_released"])
            ):
                break
        else:
            self.selected_algorithms[self.challenge_id] = "default"
        return self.selected_algorithms[self.challenge_id]

    def pickDifficultyToBenchmark(self):
        # How TIG rewards benchmarks:
        # 1. filters for benchmarks from the last 2 hours
        # 2. iteratively flag the hardest difficulty benchmarks on the 
        # pareto frontier as top performers until "target_num_solutions" is exceeded
        # 3. flag remaining benchmarks as average performers
        # 4. distribute rewards amongst top performers, pro-rata with number of solutions
        #
        # benchmarks with frontier_idx >= 0 are top performers
        # benchmarks on frontier_idx 0 represent the minimum difficulty to be considered a top performer 
        benchmarks = [
            dict(zip(self.frontiers["header_row"], row))
            for row in self.frontiers["data_rows"]
        ]

        # group benchmarks by frontier_idx
        benchmarks_by_frontier_idx = {}
        num_solutions_on_frontiers = 0
        for b in benchmarks:
            benchmarks_by_frontier_idx.setdefault(b["frontier_idx"], []).append(b)
            num_solutions_on_frontiers += b["num_solutions"] * (b["frontier_idx"] is not None)

        # FIXME this assumes Difficulty has exactly 2 parameters
        difficulty_bounds = self.round_config["difficulty_bounds"][self.challenge_id]
        x_param, y_param = list(difficulty_bounds)
        min_difficulty = {x_param: difficulty_bounds[x_param][0], y_param: difficulty_bounds[y_param][0]}
        max_difficulty = {x_param: difficulty_bounds[x_param][1], y_param: difficulty_bounds[y_param][1]}

        if 0 not in benchmarks_by_frontier_idx:
            difficulty = min_difficulty
        else:
            # randomly interpolate a point on the easiest frontier
            random_x, random_y = randomInterpolate(
                points=[
                    [b["difficulty"][x_param], b["difficulty"][y_param]] 
                    for b in benchmarks_by_frontier_idx[0]
                ],
                min_point=(min_difficulty[x_param], min_difficulty[y_param])
            )
            # randomly increment/decrement difficulty
            pos_or_neg = (-1) ** (num_solutions_on_frontiers < self.round_config["target_num_solutions"]) # hack to set True=-1, False=1
            difficulty = {
                x_param: random_x + random.randint(0, 1) * pos_or_neg,
                y_param: random_y + random.randint(0, 1) * pos_or_neg
            }

            for param in [x_param, y_param]:
                # Ensure our difficulty is within the bounds
                difficulty[param] = int(np.clip(
                    difficulty[param],
                    a_min=min_difficulty[param],
                    a_max=max_difficulty[param]
                ))
        return difficulty

    def generateAndSolveInstance(self, benchmark_params):
        Challenge = __import__(f"{self.challenge_id}.challenge").challenge.Challenge
        solveChallenge = getattr(
            __import__(f"{self.challenge_id}.algorithms.{self.algorithm_id}").algorithms,
            self.algorithm_id
        ).solveChallenge

        seed = calcSeed(**benchmark_params, nonce=self.nonce)
        try:
            start = datetime.now()
            c = Challenge.generateInstance(seed, self.difficulty)
            solution, solution_method_id = solveChallenge(c)
            if c.verifySolution(solution):
                self.proofs.append(dict(
                    nonce=self.nonce,
                    solution_base64=base64.urlsafe_b64encode(
                        minJsonDump(solution).encode()
                    ).decode(),
                    solution_method_id=int(solution_method_id) % (2 ** 31),
                    compute_time=int((datetime.now() - start).total_seconds() * 1000)
                ))
        except:
            self.num_errors += 1
        self.nonce += 1

    async def run_once(self):
        self.status = "Querying player summary"
        self.summary = await self.api.getPlayerSummary()
        await asyncio.sleep(0)

        self.status = "Querying round config"
        self.round_config = await self.api.getRoundConfig()
        for challenge_id in self.round_config["difficulty_bounds"]:
            self.selected_algorithms.setdefault(challenge_id, "default")
        await asyncio.sleep(0)

        self.status = "Querying latest block"
        self.latest_block = await self.api.getLatestBlock()
        await asyncio.sleep(0)

        self.status = "Querying player benchmarks"
        self.recent_benchmarks = await self.api.getPlayerBenchmarks()
        self.recent_benchmarks['data_rows'] += self.low_performance
        await asyncio.sleep(0)

        self.status = "Picking challenge to benchmark"
        self.challenge_id = self.pickChallengeToBenchmark()
        await asyncio.sleep(0)

        if not os.path.exists((challenge_py := f"{self.challenge_id}/challenge.py")):
            self.status = "Downloading challenge code"
            os.makedirs(f"{self.challenge_id}/algorithms")
            with open(f"{self.challenge_id}/__init__.py", "w") as f:
                pass
            challenge_code = (await request(url=f"https://raw.githubusercontent.com/the-innovation-game/challenges/{GIT_BRANCH}/{challenge_py}"))[1]
            with open(challenge_py, "w") as f:
                f.write(challenge_code)
            with open(f"{self.challenge_id}/algorithms/__init__.py", "w") as f:
                pass
            await asyncio.sleep(0)

        self.status = "Querying algorithms"
        self.all_algorithms = await self.api.getAlgorithms()
        await asyncio.sleep(0)

        self.status = "Picking algorithm to benchmark"
        self.algorithm_id = self.pickAlgorithmToBenchmark()
        await asyncio.sleep(0)

        if not os.path.exists((algorithm_py := f"{self.challenge_id}/algorithms/{self.algorithm_id}.py")):
            self.status = "Downloading algorithm code"
            algorithm_code = (await request(url=f"https://raw.githubusercontent.com/the-innovation-game/challenges/{GIT_BRANCH}/{algorithm_py}"))[1]
            with open(algorithm_py, "w") as f:
                f.write(algorithm_code)
            await asyncio.sleep(0)

        self.status = "Querying difficulty frontiers"
        self.frontiers = await self.api.getFrontiers(self.challenge_id)
        await asyncio.sleep(0)

        self.status = "Picking difficulty to benchmark"
        self.difficulty = self.pickDifficultyToBenchmark()
        await asyncio.sleep(0)

        self.benchmark_start = datetime.now().astimezone()
        self.benchmark_end = self.benchmark_start + timedelta(seconds=60)

        benchmark_params = dict(
            player_id=self.summary["player"]["id"],
            block_id=self.latest_block["block_id"],
            prev_block_id=self.latest_block["prev_block_id"],
            challenge_id=self.challenge_id,
            algorithm_id=self.algorithm_id,
            difficulty=self.difficulty
        )

        self.status = "Benchmarking"
        while (
            self.running and 
            datetime.now().astimezone() < self.benchmark_end
        ):
            if len(self.proofs) < self.round_config["max_num_solutions_per_submission"]:
                self.generateAndSolveInstance(benchmark_params)
            await asyncio.sleep(0)

        submit_status = None
        if len(self.proofs) > 0:
            self.status = "Submitting Benchmark"
            try:
                nonces = [p['nonce'] for p in self.proofs]
                solution_hashes = [calcSolutionHash(p['solution_base64']) for p in self.proofs]
                compute_times = [p.pop('compute_time') for p in self.proofs]
                resp = await self.api.submitBenchmark(
                    **benchmark_params, 
                    nonces=nonces,
                    solution_hashes=solution_hashes,
                    compute_times=compute_times
                )
                sampled_nonces = set(resp["sampled_nonces"])
                try:
                    await self.api.submitProofs(
                        resp["benchmark_id"],
                        proofs=[
                            p for p in self.proofs
                            if p['nonce'] in sampled_nonces
                        ]
                    )
                except Exception as e:
                    self.status = "Error Submitting Proofs"
            except Exception as e:
                self.status = "Error Submitting Benchmark"
                submit_status = "ERRORED"

        else:
            self.status = "Low performance. Skipping submission"
            submit_status = "LOW"

        if submit_status is not None:
            self.low_performance.append(
                (
                    str(uuid4()), # benchmark_id
                    self.benchmark_end.isoformat(), # datetime_submitted
                    0, # latest_earnings
                    self.latest_block['block_id'],
                    self.challenge_id,
                    self.algorithm_id,
                    self.difficulty,
                    0, # num_solutions
                    None, # frontier_idx
                    submit_status
                )
            )
            await asyncio.sleep(2)

        while (
            len(self.low_performance) > 0 and
            (datetime.now().astimezone() - datetime.fromisoformat(self.low_performance[0][1])).total_seconds() > self.round_config['seconds_benchmark_active']
        ):
            self.low_performance.pop(0)


    async def run_forever(self):
        while self.running:
            try:
                self.reset()
                await self.run_once()
            except Exception as e:
                self.status = f"Error: {e}"
                await asyncio.sleep(30)

G = {}

async def getBenchmarkerStatus():
    if (b := G.get("benchmarker")) is None:
        return dict(running=False)
    else:
        return dict(
            running=True,
            current_benchmark=dict(
                status=b.status,
                start=None if b.benchmark_start is None else b.benchmark_start.isoformat(),
                end=None if b.benchmark_end is None else b.benchmark_end.isoformat(),
                challenge_id=b.challenge_id,
                algorithm_id=b.algorithm_id,
                difficulty=b.difficulty,
            ),
            round_config=b.round_config,
            summary=b.summary,
            all_algorithms=b.all_algorithms,
            selected_algorithms=b.selected_algorithms,
            recent_benchmarks=b.recent_benchmarks,
        )

async def startBenchmarker(player_id, api_key):
    if (b := G.get("benchmarker")) is None:
        b = SimpleBenchmarker(
            API(API_URL, api_key, player_id)
        )
        G["benchmarker"] = b
        await b.run_forever()

async def stopBenchmarker():
    if (b := G.pop("benchmarker")) is not None:
        b.running = False
        
async def selectAlgorithm(challenge_id, algorithm_id):
    if (b := G.get("benchmarker")) is not None:
        b.selected_algorithms.update({challenge_id: algorithm_id})