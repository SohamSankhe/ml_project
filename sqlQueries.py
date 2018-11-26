

MATCHES = '''SELECT match_id FROM playerperformance p group by match_id;'''
             #limit 500;'''

PLAYER_TEAM_RATINGS = '''select p.match_id, q.overall as 'player_rating' , t.team_rating, p.team_id from
    playerperformance p left join playermatchstatstable q on p.player_name = q.player_name
    left join teamdetails t on p.match_id = t.match_id and p.team_id = t.team_id
    where p.match_id = {match_identifier} and player_position_info not like '%sub%' order by p.match_id, p.team_id'''
    # limit 30;'''  # remove limit later


MATCH_FEATURES_TRAINING = '''SELECT
    tr.match_id, tr.home_team_id, tr.away_team_id, tr.home_team_rating, tr.away_team_rating,
    tp.goal_diff_home,tp.home_win_perc,tp.home_lose_perc,tp.home_draw_perc,
    tpa.goal_diff_away,tpa.away_win_perc,tpa.away_lose_perc,tpa.away_draw_perc,
    s.full_time_score,
    (if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) > SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 1,
     (if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) = SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 0, -1)))) 
     as 'match_outcome'
    FROM teamratings tr
    inner join
    teamperformance tp on tr.home_team_id = tp.team_id
    inner join
    teamperformance tpa on tr.away_team_id = tpa.team_id
    inner join
    season_match_stats s on tr.match_id = s.match_id and tr.home_team_id = s.home_team_id
    where tr.match_id in ({match_identifier})
    order by tr.match_id, tr.home_team_id
    ;'''

MATCH_FEATURES_TESTING = '''SELECT
    tr.match_id, tr.home_team_id, tr.away_team_id, 0.000 as 'home_team_rating', 0.000 as 'away_team_rating',
    tp.goal_diff_home,tp.home_win_perc,tp.home_lose_perc,tp.home_draw_perc,
    tpa.goal_diff_away,tpa.away_win_perc,tpa.away_lose_perc,tpa.away_draw_perc,
    s.full_time_score,
    (if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) > SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 1,
     (if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) = SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 0, -1)))) 
     as 'match_outcome'
    FROM teamratings tr
    inner join
    teamperformance tp on tr.home_team_id = tp.team_id
    inner join
    teamperformance tpa on tr.away_team_id = tpa.team_id
    inner join
    season_match_stats s on tr.match_id = s.match_id and tr.home_team_id = s.home_team_id
    where tr.match_id in ({match_identifier})
    order by tr.match_id, tr.home_team_id
    ;'''