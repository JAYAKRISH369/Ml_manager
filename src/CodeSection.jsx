import "./CodeSection.css";
// import {useRef, useEffect} from 'react';
function CodeSection(props){
  // const codeRef=useRef(null);
  // let Content;
  // useEffect(()=>{
  //   if(codeRef.current){
  //      Content=codeRef.current.innerText;
  //   }

  // },[]);
  // function copyToClipboard() {
  //   var tempInput = document.createElement("input");
  //   tempInput.value = Content;
  //   document.body.appendChild(tempInput);
  //   tempInput.select();
  //   document.execCommand("copy");
  //   document.body.removeChild(tempInput);
  
  //   alert("Content copied to clipboard!");
  // }
// section div onClick={copyToClipboard}
// code div ref={codeRef}
  return(
    <div className="Section" >
<div className="Title">
{` ${props.name}`}
</div>
{/* <div className="Code" ref={codeRef}>
{props.code}
</div> */}
<pre className="Code" >
  {props.code}
</pre>
    </div>
  );
}
export default CodeSection;

////