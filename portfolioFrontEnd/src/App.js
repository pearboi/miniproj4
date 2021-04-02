import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import TextField from '@material-ui/core/TextField';
import React, {Component} from 'react';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';

export default class App extends Component{
  constructor(props){
    super(props)
    this.state = {
      cases: null,
      days: null,
      prediction: null
    };
    this.handleChange = this.handleChange.bind(this);
    this.queryModelPrediction = this.queryModelPrediction.bind(this);
    this.updatePrediction = this.updatePrediction.bind(this);
  }

  handleChange({target}){
    this.setState({
      [target.name]: target.value 
    })
  }
  
  updatePrediction(pred){
    this.setState({
      prediction: pred
    })
  }

  queryModelPrediction(){
    console.log(this.state.cases)
    console.log(this.state.days)
    this.setState({prediction: 'Loading...'})
    axios.post('http://localhost:8000/model', {
      params: {
        cases: this.state.cases,
        days: this.state.days,
      }
  }).then( (response) => {
    console.log(response);
    console.log(response['data'])
    console.log(response.data)
    this.setState({prediction: response.data})
  })
  }
  render(){
    return(
      <div>
        <Grid container spaceing={3}>
          <Box> </Box>
          <Grid item xs={2}>
            <TextField id="outlined-basic" name='cases' label="Number of Cases" xs='3' variant="outlined" value={this.state.cases} onChange={this.handleChange} />
          </Grid>
          <Grid item xs={2}>
            <TextField id="outlined-basic" name='days' label="Days Elapsed" xs='3' variant="outlined" value={this.state.days} onChange={this.handleChange} />
          </Grid>
          <Grid item xs={2}>
            <Button variant="contained" style={{backgroundColor: '#ff9100'}} className={'button'} padding={10} xs={12} onClick={() => {this.queryModelPrediction()}}>Predict</Button>
          </Grid>
          <Grid item xs = {2}>
            {this.state.prediction}
          </Grid>
        </Grid>
      </div>
      
    )
  }
}
