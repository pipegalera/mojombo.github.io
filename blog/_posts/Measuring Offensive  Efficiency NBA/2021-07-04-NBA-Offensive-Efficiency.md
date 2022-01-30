---
layout: post
title: 'Measuring Efficient Offensive Production in the NBA'
---

 {{ page.title }}
================

More important than how a player chooses to score is whether or not his scoring is efficient. No one ruins a pick-up game like the gunner, throwing bricks every two possessions. Scoring more points does not necessarily help a team. Scoring 30 points in 20 attempts is great for the team, but doing it in 70 attempts is terrible. Players cannot waste possessions, the scoring must be efficient.

The goal of the post is **building a metric of offensive efficiency** that can be used to detect winning players and teams. I believe that efficient players make for efficient teams, and efficient teams win.

I will develop two individual offensive measures: Offensive Efficiency and Efficient Offensive Production.

As the name says, **Offensive Efficiency (OE)** is a measure of how efficient is the offense of a player. A player that has basic decision-making skills and production efficiency will have high offensive efficiency. OE is a quality metric, it doesn't take into account if the player scored 3 points or 30 points, only how efficient that was.

We want to include OE in a metric that considers both quality and quantity of production.

**Efficient Offensive Production (EOP)** takes the offensive efficiency and incorporates the amount of offensive production of a player. The metric multiplies the points and assists scored by the OE coefficient to count how efficient the offensive production was.


## Offensive Efficiency (OE)

Offensive Efficiency (OE) describes the rate of successful offensive possessions the player was directly involved.

$$
OE_{i,s}= \frac{FG_i+APG_i}{FGA_i-ORB_i+APG_i+TO_i}
$$

The Offensive Efficiency of a player $i$ in the season $s$ is the number of successful $i$ player offensive possessions per game -Field Goal and Assist- divided by the player's number of potential ends of possession per game -Field Goal Attempt, Assist, Turn Over or Offensive Rebound-.

**The score** represents the performing efficiency. If a player has an OE of 0.6, we might think of that player as performed at 60% of excellent efficiency. Meaning that in 60% of the possessions that he was directly involved, with this player assisting or scoring. Players with non-marginal minutes have an Offensive Efficiency coefficient between 0 and 1.

- **A OE score of 0** means that a player did not make any field goals or assists, and therefore his team finished the possession without any basket that can be attributed to him.

- **A OE score of 1** means that the player is 100% efficient, all his points and plays are efficient. A video game NBA 2K player, every play that he runs ends with an assist or score.

While there are [players that contribute to the offense of the game without assisting or scoring](https://www.nytimes.com/2009/02/15/magazine/15Battier-t.html), for example moving the ball or creating spaces for your teammates, we tend to value the offense of a player when it is translated to points.

This calculation can be done in Python, and it returns the same dataset with an added column measuring offensive efficiency:

```python
def offensive_efficiency(df):
    df["OE"] = (df.FG + df.AST) / (df.FGA - df.ORB + df.AST + df.TOV)
    df["OE"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df
```

## Offensive Efficiency among Elite Players

As you can imagine, this metric rewards a player that creates points. The following table presents the top 20 players in points per game for the 2021 season sorted by their OE.


**Top 20 PPG Players ranked by their Offensive Efficiency for the 2021 Season**


|Rank| Player                |   OE |   PPG | 
|----|-----------------------|------|-------|
|  1 | Nikola Jokić          | 0.7  |  26.4 | 
|  2 | Zion Williamson       | 0.68 |  27   | 
|  3 | Giannis Antetokounmpo | 0.63 |  28.1 | 
|  4 | Kawhi Leonard         | 0.6  |  24.8 |
|  5 | Kyrie Irving          | 0.59 |  26.9 | 
|  5 | LeBron James          | 0.59 |  25   | 
|  7 | Kevin Durant          | 0.58 |  26.9 | 
|  7 | Karl-Anthony Towns    | 0.58 |  24.8 | 
|  9 | De'Aaron Fox          | 0.57 |  25.2 | 
| 10 | Luka Dončić           | 0.56 |  27.7 |
| 10 | Trae Young            | 0.56 |  25.3 |
| 12 | Damian Lillard        | 0.55 |  28.8 |
| 12 | Joel Embiid           | 0.55 |  28.5 |
| 14 | Zach LaVine           | 0.54 |  27.4 |
| 15 | Stephen Curry         | 0.53 |  32   |
| 15 | Bradley Beal          | 0.53 |  31.3 |
| 18 | Devin Booker          | 0.52 |  25.6 |
| 19 | Jayson Tatum          | 0.51 |  26.4 |
| 19 | Donovan Mitchell      | 0.51 |  26.4 |


You might notice that players that score near the rim have a better OE, which makes sense as near the rim is the most efficient spot in the game[^1].

**The average Offensive Efficiency for this season was 0.57**. The 40% of the top scorers are more efficient than the league average, while others the resting 60% are not. The next section explains more deeply how some of these players could be more efficient but are forced to make tough shots.

## Making sense of the team context evaluating individual Offensive Efficiency

An Offensive Efficiency below the League average doesn't imply that a player is a below-average offensive player. **The context of this metric matters**. The more a player is asked to do, the harder it becomes to produce with high efficiency.

For example, Stephen Curry was asked to do a lot this 2021 season to carry the offense of the Warriors. **Good players in bad teams are likely to score less efficiently** because they have to score more for the team, and in more difficult situations. Players like Steph can be more efficient, but their OE is biased down because of their team context.

Golden State's offense gravitates towards **Stephen Curry**. Opposing teams tighten spaces around him and constantly dropping bodies if he tries to penetrate the basket. As a result, Curry is forced to make difficult shots.

In contrast, **Kyle Irving** can get more assists, open floors to dribble, and space to score just because he has elite teammates. **His teammates make him more efficient individually**. Irving can easily dribble to create some space for the jump shots because the defenders have to stick to their marks. 2 against 1 is not an option when you have to defend also James Harden, Kevin Durant and Joe Harris.

**Team context plays an enormous role in players efficiency stats and therefore bias Offensive Efficiency.**

From the volume statistical perspective, having good teammates have a disadvantage. Irving has better OE but also has to share the field goals and assists with Durant and Harden. He scored 26.9 PPG and 6 AST. Stephen curry scores less efficiently, but also has more attempts to make shots. He scored 32 PGG and 5.8 AST.**Good players in bad teams are likely to score more points**.

Once we move deeper into a metric that counts the production **amount** among with the efficiency, Curry's extra buckets attenuate the lack of efficiency because of his *team-context*.

## Offensive Efficiency tale: LeBron and Carmelo

We can compare the evolution of Offensive Efficiency of different players over time, to see if they could be a **correlation of efficiency and team wins**.

Let's go back to the 2003 NBA draft, [one of the greatest drafts of all time](https://bleacherreport.com/articles/2784908-ranking-the-top-10-nba-draft-classes-of-all-time). The first 5 picks includes LeBron James, Dwyane Wade, Chris Bosh, and Carmelo Anthony. All of them are or will be [Hall of Famers](https://www.basketball-reference.com/leaders/hof_prob.html). I will compare the best two scorers of the draft to show why efficiency matters and how much. These two players are Lebron James and Carmelo Anthony.

![](https://media.bleacherreport.com/w_768,h_512,c_fill/br-img-images/002/372/680/ScreenShot2013-06-25at12.39.15PM_crop_north.jpg)

They are 2 of the best players in history when it comes to quantity of offensive production, and the only two active players in the [Top 25 NBA History Points Leaders](http://www.espn.com/nba/history/leaders). However, one of them translated the points into rings and the other did not. Did efficiency play a role in this?

The following graph shows the OE in the **regular season** of both players over time:

![](/images/blogs/lebron_carmelo.png)

**LeBron efficiency peaks in the 2012-2013 season with Miami**. They won 27 games in a row that season (the second-longest winning streak in NBA history), finished with a 66–16 record. LeBron's most efficient regular season was translated into an NBA championship for the Miami Heat, being the MVP of both the regular season and the finals.

That same season Carmelo Anthony was the PPG league leader with 28.7 PGG, but the Knicks faded in the conference semifinals against the Indiana Pacers. **The main names of New York were not that far in terms of points per game from Miami's big three**: Carmelo, JR Smith, and Stoudemire together averaged 61 PGG against the 64.6 PGG of LeBron, Wade, and Bosh. JR at the time was a top 20 league scorer. But in terms of efficiency, they were worlds apart.

**Maybe a better team cast would have helped Melo to be more efficient**, but it seems unlikely looking at the offensive efficiency playing in different teams. After being the roster star of the Knicks, he jumped in between teams with little success. People were speculating that he would retire from professional basketball after spending a year without a team. Fortunately, Melo joined the Blazers in 2020 with a **one-year non-guaranteed deal**.

As *Sports Illustrated* put in [an article about his journey](https://www.gq.com/story/who-is-carmelo-anthony-now): 

<figure class="quote">
  <blockquote>
This [Blazers] Melo is about efficiency and sacrifice: Anthony would primarily look to melt defenses behind the three-point line, cure himself of all urges to go one-on-one, and give consistent effort on defense.
  </blockquote>
  <figcaption>
    &mdash; Michael Pina, <cite>Sports Illustrated (August 20, 2020)</cite>  </figcaption>
</figure>


It is safe to say that if Melo hasn't changed to a more efficient style he likely would be out of the NBA.

Simply put, Carmelo can score more overall buckets but LeBron scores more buckets per possession. LeBron is more efficient, a better offensive player. And as I said at the beginning of the post: **efficient players make for efficient teams, and efficient teams win**.


## The value of an assist

We talked about how centers can be efficient because they score mostly around the rim, but what about the teammates that set up the dunks and layouts? Players should be rewarded for their assists. Before we create the Efficient Offensive Production (EOP) metric, we need to define the value of an assist so we can reward good passers.

[A recent paper](https://arxiv.org/pdf/1902.08081.pdf) shows that 72% of assists increase the expected points of a field goal attempt. However, how many expected points? In other words, **how much is an assist worth relative to a point scored?** I am not going to drive deep into the assist value, but an easy answer would be **it depends**.

Compare these 2 situations:

- Campazzo passes the ball to Jokić. Jokić pushes the defender and ends with a hook near the rim. Jokić scores, Campazzo assists.
- Tatum iso against his defender. He quickly penetrates to the rim driving 2 more defenders to him. Surrounded by defenders passes the ball to Ojeleye at the left corner that takes the open shot. Ojeleye scores, Tatum assists.

If you are a Nuggets or Celtics fan, you must have seen these situations plenty of times this season. The result is the same but the value of these assists are quite different.

![Boston Celtics forward Jayson Tatum, center, drives to the rim as, Denver Nuggets defenders Paul Millsap, left, and Michael Porter Jr., right, defend in the first half of an NBA basketball game Sunday, April 11, 2021, in Denver. (AP Photo/David Zalubowski)AP](https://www.masslive.com/resizer/qNPoaECxb0ZVNCwI3NUb-zdqgiU=/1280x0/smart/cloudfront-us-east-1.images.arcpublishing.com/advancelocal/SSUJAYXBYFA4NEBFLF35Z4J7AM.jpg)

In theory, we could calculate each assist worth according to the extra expected points added and where the scorer was, for each assist. For example, assisted corner threes would worth more than simple assists moving the ball at the perimeter. But this approach would require spatial-temporal data and breaking a rule for this metric: keep it simple.

To keep the assist value wrapped in a simple coefficient, I will take **the average expected points added from an assist** stated in [A. Sicilia et al. (2019)](https://arxiv.org/pdf/1902.08081.pdf). On average, **an assisted shot added 0.16 expected points more compared to an unassisted shot**. This result is more accurate and significantly lower than the previous coefficient of 0.76 from the first attempts to calculate the assist value[2].

While this "shortcut" is not perfect, it allows data and NBA enthusiasts like me to being able to [download free aggregate data](https://share.streamlit.io/pipegalera/basketballreference-webscraper/main) and building this metric.

## Efficient Offensive Production (EOP)

Now we know how to weight assists, we can move towards a complete offensive metric.

Recall that **Offensive Efficiency** only measures individual **quality** offense. We introduce Efficient Offensive Production (EOP) to **incorporate the amount of production** a player provides.

I define Efficient Offensive Production as:

$$
EOP_{i,s} = [(0.16 \times APG_{i,s}) + PPG_{i,s})] \times OE_{i}
$$

Efficient Offensive Production (EOP) of player $$i$$ at season $$s$$ is equal to the sum of his points and weighted assist times his efficiency parameter OE.

The application in Python follows as:

```python
def EOP(df):
    df["EOP"] = [(0.16 * df["AST"] + df["PTS"])] * [(df.FG + df.AST) / (df.FGA - df.ORB + df.AST + df.TOV)]
    df["EOP"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df
```

The following table presents the top 20 players in points per game for the 2021 season sorted by their EOP.

**Top 20 PPG Players ranked by their Efficient Offensive Production for the 2021 Season**

|Rank| Player                | EOP | PPG |
|----|-----------------------|-------|-------|
|  1 | Nikola Jokić          | 19.28 |  26.4 |
|  2 | Zion Williamson       | 18.79 |  27   |
|  3 | Giannis Antetokounmpo | 18.31 |  28.1 |
|  4 | Stephen Curry         | 17.55 |  32   |
|  5 | Bradley Beal          | 17.04 |  31.3 |
|  6 | Damian Lillard        | 16.56 |  28.8 |
|  7 | Luka Dončić           | 16.41 |  27.7 |
|  7 | Kyrie Irving          | 16.41 |  26.9 |
|  9 | Kevin Durant          | 16.05 |  26.9 | 
| 10 | Joel Embiid           | 16.04 |  28.5 | 
| 11 | LeBron James          | 15.46 |  25   |
| 12 | Kawhi Leonard         | 15.31 |  24.8 |
| 13 | Zach LaVine           | 15.23 |  27.4 |
| 14 | Trae Young            | 14.98 |  25.3 |
| 15 | De'Aaron Fox          | 14.97 |  25.2 |
| 16 | Karl-Anthony Towns    | 14.74 |  24.8 |
| 17 | Donovan Mitchell      | 13.96 |  26.4 |
| 18 | Jayson Tatum          | 13.95 |  26.4 |
| 19 | Devin Booker          | 13.7  |  25.6 |
| 20 | Jaylen Brown          | 13.3  |  24.7 |

The way the value should be interpreted is as efficient points made. In the end, EOP is points plus assists "translated to points" multiplied by the players' efficiency coefficient OE. Not surprisingly **Nikola Jokić ranks first**, as he is a prolific passer, a very efficient scorer, and a volume shooter.

There are significant differences in rankings according to EOP and OE. EOP rewards more fairly high volume three-pointers and volume scorers like Bradley Beal and gives more credit to players that drive team offense like Stephen Curry and Luka Dončić. It details better than OE the total offensive contribution of a player as it accounts for the number of points produced and includes the value of assists. 

[^1]: Shea, Stephen M.: [ShotTracker.com - Analytics & Shot Selection](https://shottracker.com/articles/analytics-shot-selection)
[^2]: Shea, Stephen M., and Christopher E. Baker. Basketball analytics: Objective and efficient strategies for understanding how teams win. Advanced Metrics, 2013.
